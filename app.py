import streamlit as st
import os
import faiss
import numpy as np
import tempfile
import uuid

import vertexai
from vertexai.vision_models import (
    MultiModalEmbeddingModel,
    Image,
    Video,
    VideoSegmentConfig,
)
from vertexai.generative_models import GenerativeModel, Part

from google.cloud import storage
from google.oauth2 import service_account

from pypdf import PdfReader
from docx import Document


# ==========================================
# CONFIG
# ==========================================

PROJECT_ID = "video-embedding-488510"
LOCATION = "us-central1"
BUCKET_NAME = "multimodal_rag_01"
EMBED_DIM = 1408

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials,
)

embedding_model = MultiModalEmbeddingModel.from_pretrained(
    "multimodalembedding@001"
)

gemini_model = GenerativeModel("gemini-2.5-flash")


# ==========================================
# SESSION STATE
# ==========================================

if "index" not in st.session_state:
    st.session_state.index = faiss.IndexFlatIP(EMBED_DIM)
    st.session_state.metadata = []

def normalize(vec):
    return vec / np.linalg.norm(vec)

def store_vector(vec, metadata):
    vec = normalize(vec)
    st.session_state.index.add(np.array([vec]).astype("float32"))
    st.session_state.metadata.append(metadata)


# ==========================================
# GCS UPLOAD
# ==========================================

def upload_to_gcs(local_path, filename):
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(BUCKET_NAME)
    blob_name = f"uploads/{uuid.uuid4()}_{filename}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET_NAME}/{blob_name}"


# ==========================================
# TEXT PROCESSING
# ==========================================

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def embed_text_chunks(text, source, max_chars=900):
    start = 0
    while start < len(text):
        chunk = text[start:start+max_chars]
        emb = embedding_model.get_embeddings(contextual_text=chunk)
        vec = np.array(emb.text_embedding, dtype="float32")

        store_vector(vec, {
            "type": "text",
            "content": chunk,
            "source": source
        })

        start += max_chars


# ==========================================
# IMAGE EMBEDDING
# ==========================================

def embed_image(path):
    image = Image.load_from_file(path)
    emb = embedding_model.get_embeddings(image=image)
    vec = np.array(emb.image_embedding, dtype="float32")

    store_vector(vec, {
        "type": "image",
        "path": path
    })


# ==========================================
# VIDEO EMBEDDING (SEGMENTS)
# ==========================================

def embed_video(gcs_uri):
    video = Video.load_from_uri(gcs_uri)

    emb = embedding_model.get_embeddings(
        video=video,
        video_segment_config=VideoSegmentConfig(
            start_offset_sec=0,
            end_offset_sec=60,
            interval_sec=10,
        ),
    )

    for seg in emb.video_embeddings:
        vec = np.array(seg.embedding, dtype="float32")

        store_vector(vec, {
            "type": "video",
            "gcs_uri": gcs_uri,
            "start": seg.start_offset_sec,
            "end": seg.end_offset_sec,
        })


# ==========================================
# VIDEO TRANSCRIPTION (STREAMING)
# ==========================================

def transcribe_video(gcs_uri):
    response = gemini_model.generate_content(
        [
            "Transcribe this video accurately in English with speaker diarization.",
            Part.from_uri(gcs_uri, mime_type="video/mp4"),
        ],
        stream=True,
    )

    transcript = ""
    placeholder = st.empty()

    for chunk in response:
        if chunk.text:
            transcript += chunk.text
            placeholder.markdown(transcript + "▌")

    placeholder.markdown(transcript)
    return transcript


# ==========================================
# SUMMARY (STREAMING)
# ==========================================

def summarize(text):
    response = gemini_model.generate_content(
        f"Provide a concise summary:\n{text}",
        stream=True,
    )

    summary = ""
    placeholder = st.empty()

    for chunk in response:
        if chunk.text:
            summary += chunk.text
            placeholder.markdown(summary + "▌")

    placeholder.markdown(summary)
    return summary


# ==========================================
# SEARCH
# ==========================================

def search(query, k=3):
    if st.session_state.index.ntotal == 0:
        return []

    emb = embedding_model.get_embeddings(contextual_text=query)
    query_vec = normalize(np.array(emb.text_embedding, dtype="float32"))

    distances, indices = st.session_state.index.search(
        np.array([query_vec]).astype("float32"), k
    )

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append(st.session_state.metadata[idx])

    return results


# ==========================================
# HYBRID CHAT
# ==========================================

def answer_query(query, item):

    prompt_parts = [query]

    if item["type"] == "text":
        prompt_parts.append(f"\nContext:\n{item['content']}")

    elif item["type"] == "image":
        prompt_parts.append(
            Part.from_data(
                data=open(item["path"], "rb").read(),
                mime_type="image/jpeg"
            )
        )

    elif item["type"] == "video":
        prompt_parts.append(
            Part.from_uri(item["gcs_uri"], mime_type="video/mp4")
        )

    response = gemini_model.generate_content(prompt_parts, stream=True)

    full_response = ""
    placeholder = st.empty()

    for chunk in response:
        if chunk.text:
            full_response += chunk.text
            placeholder.markdown(full_response + "▌")

    placeholder.markdown(full_response)


# ==========================================
# STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("🚀 Full Multimodal RAG System")

uploaded = st.file_uploader(
    "Upload PDF / DOCX / Image / Video",
    type=["pdf", "docx", "jpg", "png", "jpeg", "mp4"]
)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
        path = tmp.name

    ext = uploaded.name.split(".")[-1]

    if ext == "pdf":
        text = extract_text_from_pdf(path)
        embed_text_chunks(text, uploaded.name)
        st.success("PDF Embedded!")

    elif ext == "docx":
        text = extract_text_from_docx(path)
        embed_text_chunks(text, uploaded.name)
        st.success("DOCX Embedded!")

    elif ext in ["jpg", "png", "jpeg"]:
        st.image(uploaded)
        embed_image(path)
        st.success("Image Embedded!")

    elif ext == "mp4":
        st.video(uploaded)

        gcs_uri = upload_to_gcs(path, uploaded.name)
        embed_video(gcs_uri)

        st.subheader("Transcript")
        transcript = transcribe_video(gcs_uri)

        st.subheader("Summary")
        summary = summarize(transcript)

        embed_text_chunks(transcript, uploaded.name)
        st.success("Video Processed Successfully!")


st.header("🔎 Semantic Search")

search_query = st.text_input("Search")

if st.button("Search") and search_query:
    results = search(search_query, k=3)
    for result in results:
        st.write(result)


st.header("💬 Chat with Your Data")

query = st.text_input("Ask a question")

if st.button("Ask") and query:
    results = search(query, k=1)
    if not results:
        st.warning("No relevant data found.")
    else:
        answer_query(query, results[0])

st.divider()
st.write(f"Vector DB Size: {len(st.session_state.metadata)}")