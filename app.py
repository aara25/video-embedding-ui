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
chat_model = init_chat_model("gemini-2.5-flash")

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

def embed_video(path):
    video = Video.load_from_file(path)

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
            "path": path,
            "start": seg.start_offset_sec,
            "end": seg.end_offset_sec,
        })

# -----------------------------------
# VIDEO TRANSCRIPT
# -----------------------------------

def transcribe_video(path):
    with open(path, "rb") as f:
        video_bytes = f.read()

    response = gemini_model.generate_content([
        "Transcribe the speech in this video accurately in english with user diarriazation. "
        "Return only the spoken words.",
        Part.from_data(data=video_bytes, mime_type="video/mp4"),
    ])

    return response.text

# -----------------------------------
# SUMMARY
# -----------------------------------

def summarize(text):
    response = gemini_model.generate_content(
        f"Give a concise summary of this:\n{text}"
    )
    return response.text

# -----------------------------------
# STREAMLIT UI
# -----------------------------------

st.set_page_config(page_title="Vertex Multimodal RAG", layout="wide")
st.title("🚀 Gemini Vertex Multimodal RAG")

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
        embed_video(path)
        transcript = transcribe_video(path)
        summary = summarize(transcript)

        st.subheader("Transcript")
        st.write(transcript)

        st.subheader("Summary")
        st.write(summary)

        embed_text_chunks(transcript, uploaded.name)
        st.success("Video Embedded + Transcript Stored!")

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

if st.button("Ask"):
    results = search(query)

    if not results:
        st.warning("No relevant results found.")
    else:
        context = ""
        for r in results:
            if r["data"]["type"] == "text":
                context += r["content"] + "\n"
            elif r["data"]["type"] == "image":
                context += f"Image file: {r['path']}\n"
            elif r["data"]["type"] == "video":
                context += f"Video segment {r['start']}-{r['end']} sec\n"

        response = chat_model.invoke(
            [HumanMessage(content=f"Context:\n{context}\n\nQuestion:{query}")]
        )

        st.write("### 🧠 Answer")
        st.write(response.content)

st.divider()
st.write(f"Vector DB Size: {len(st.session_state.metadata)}")