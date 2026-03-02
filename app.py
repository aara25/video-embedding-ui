import streamlit as st
import os
import faiss
import numpy as np
import tempfile

import vertexai
from vertexai.vision_models import (
    MultiModalEmbeddingModel,
    Image,
    Video,
    VideoSegmentConfig,
)
from vertexai.generative_models import GenerativeModel, Part

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pypdf import PdfReader
from docx import Document

from google.cloud import storage


# -----------------------------------
# CONFIG
# -----------------------------------

from google.oauth2 import service_account


credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

PROJECT_ID = "video-embedding-488510"
LOCATION = "us-central1"
EMBED_DIM = 1408

vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

embedding_model = MultiModalEmbeddingModel.from_pretrained(
    "multimodalembedding@001"
)
gemini_model = GenerativeModel("gemini-2.5-flash")
chat_model = GenerativeModel("gemini-2.5-flash")

def upload_to_gcs(local_path, bucket_name, blob_name):
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

    return f"gs://{bucket_name}/{blob_name}"

# -----------------------------------
# SESSION STATE
# -----------------------------------

if "index" not in st.session_state:
    st.session_state.index = faiss.IndexFlatIP(EMBED_DIM)
    st.session_state.metadata = []

def normalize(vec):
    return vec / np.linalg.norm(vec)

def store_vector(vec, metadata):
    vec = normalize(vec)
    st.session_state.index.add(np.array([vec]).astype("float32"))
    st.session_state.metadata.append(metadata)

def search(query, k=3):
    if st.session_state.index.ntotal == 0:
        return []

    emb = embedding_model.get_embeddings(contextual_text=query)
    query_vec = normalize(np.array(emb.text_embedding, dtype="float32"))

    distances, indices = st.session_state.index.search(
        np.array([query_vec]).astype("float32"), k
    )

    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(st.session_state.metadata[idx])
    return results

# -----------------------------------
# DOCUMENT PROCESSING
# -----------------------------------

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
    """
    Splits text safely under Vertex 1024 char limit.
    Keeps buffer at 900 for safety.
    """
    start = 0
    text_length = len(text)

    while start < text_length:
        chunk = text[start:start + max_chars]

        emb = embedding_model.get_embeddings(contextual_text=chunk)
        vec = np.array(emb.text_embedding, dtype="float32")

        store_vector(vec, {
            "type": "text",
            "content": chunk,
            "source": source
        })

        start += max_chars

# -----------------------------------
# IMAGE EMBEDDING
# -----------------------------------

def embed_image(path):
    image = Image.load_from_file(path)
    emb = embedding_model.get_embeddings(image=image)
    vec = np.array(emb.image_embedding, dtype="float32")

    store_vector(vec, {
        "type": "image",
        "path": path
    })

# -----------------------------------
# VIDEO EMBEDDING (SEGMENT LEVEL)
# -----------------------------------

def embed_video(gcs_uri):
    # Load video directly from GCS
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

# -----------------------------------
# VIDEO TRANSCRIPT
# -----------------------------------
def transcribe_video_gcs(gcs_uri):
    response = gemini_model.generate_content(
        [
            "Transcribe this video accurately in English with diarization.",
            Part.from_uri(gcs_uri, mime_type="video/mp4"),
        ],
        stream=True,
    )

    transcript = ""
    for chunk in response:
        if chunk.text:
            transcript += chunk.text
            st.write(chunk.text)

    return transcript

# -----------------------------------
# SUMMARY
# -----------------------------------

def summarize(text):
    response_stream = gemini_model.generate_content(
        f"Give a concise summary of this:\n{text}",
        stream=True,
    )

    summary = ""
    placeholder = st.empty()

    for chunk in response_stream:
        # Some chunks may not contain text
        if hasattr(chunk, "text") and chunk.text:
            summary += chunk.text
            placeholder.markdown(summary + "▌")

    # Remove cursor after completion
    placeholder.markdown(summary)

    return summary

# -----------------------------------
# STREAMLIT UI
# -----------------------------------

st.set_page_config(page_title="Vertex Multimodal RAG", layout="wide")
st.title("Multimodal RAG")

uploaded = st.file_uploader(
    "Upload PDF / DOCX / Image / Video",
    type=["pdf", "docx", "jpg", "png", "jpeg", "mp4"]
)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
        path = tmp.name

    ext = uploaded.name.split(".")[-1]

    # ---------------- PDF ----------------
    if ext == "pdf":
        with st.spinner("Extracting and embedding PDF..."):
            text = extract_text_from_pdf(path)
            embed_text_chunks(text, uploaded.name)

        st.success("PDF Embedded Successfully!")

    # ---------------- DOCX ----------------
    elif ext == "docx":
        with st.spinner("Extracting and embedding DOCX..."):
            text = extract_text_from_docx(path)
            embed_text_chunks(text, uploaded.name)

        st.success("DOCX Embedded Successfully!")

    # ---------------- IMAGE ----------------
    elif ext in ["jpg", "png", "jpeg"]:
        st.image(uploaded)

        with st.spinner("Generating image embedding..."):
            embed_image(path)

        st.success("Image Embedded Successfully!")

    # ---------------- VIDEO ----------------
    elif ext == "mp4":
        st.video(uploaded)

        # ---------------- Upload to GCS ----------------
        with st.spinner("Uploading video to cloud storage..."):
            blob_name = f"videos/{uploaded.name}"
            gcs_uri = upload_to_gcs(path, "multimodal_rag_01", blob_name)


        # Step 2: Transcript
        with st.spinner("Transcribing video..."):
            transcript = transcribe_video_gcs(gcs_uri)

        # Step 3: Summary
        with st.spinner("Generating summary..."):
            summary = summarize(transcript)

        st.subheader("Summary")
        st.write(summary)

        # Step 4: Store transcript embeddings
        with st.spinner("Embedding transcript for semantic search..."):
            embed_text_chunks(transcript, uploaded.name)

        st.success("✅ Video Processed & Embedded Successfully!")

def search(query, k=5):
    if st.session_state.index.ntotal == 0:
        return []

    # Embed query
    emb = embedding_model.get_embeddings(contextual_text=query)
    query_vec = normalize(np.array(emb.text_embedding, dtype="float32"))

    # Search FAISS
    distances, indices = st.session_state.index.search(
        np.array([query_vec]).astype("float32"), k
    )

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx != -1:
            item = st.session_state.metadata[idx]
            results.append({
                "score": float(score),
                "data": item
            })

    return results

# -----------------------------------
# CHAT
# -----------------------------------

st.header("🔎 Semantic Search")

search_query = st.text_input("Search your knowledge base")

if st.button("Search"):
    results = search(search_query, k=5)

    if not results:
        st.warning("No results found.")
    else:
        for i, result in enumerate(results, 1):
            st.markdown(f"### Result {i}")
            st.write(f"Similarity Score: {result['score']:.4f}")

            data = result["data"]

            if data["type"] == "text":
                st.write(data["content"])

            elif data["type"] == "image":
                st.write("Image File:", data["path"])

            elif data["type"] == "video":
                st.write(
                    f"Video Segment: {data['start']}s - {data['end']}s"
                )

            st.divider()

st.header("💬 Chat with Your Data")

query = st.text_input("Ask something...")

if st.button("Ask") and query:

    retrieved = search(query, k=1)

    if not retrieved:
        st.warning("I haven't learned anything yet! Please upload content first.")
    else:
        item = retrieved[0]["data"]

        context = ""

        # -------- TEXT --------
        if item["type"] == "text":
            context = item["content"]

        # -------- IMAGE --------
        elif item["type"] == "image":
            context = f"This question relates to the image stored at {item.get('path','')}."

        # -------- VIDEO --------
        elif item["type"] == "video":
            context = (
                f"This question relates to video segment "
                f"{item['start']}s to {item['end']}s "
                f"from {item.get('gcs_uri','')}."
            )

        # -------- STREAM CHAT RESPONSE --------
        with st.container():
            st.subheader("🧠 Answer")

            response_stream = gemini_model.generate_content(
                f"Context:\n{context}\n\nQuestion:{query}",
                stream=True
            )

            full_response = ""
            placeholder = st.empty()

            for chunk in response_stream:
                if hasattr(chunk, "text") and chunk.text:
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

st.divider()
st.write(f"Vector DB Size: {len(st.session_state.metadata)}")