import os
import time
import requests
import streamlit as st

from rag_pipeline.loader import load_documents, split_documents
from rag_pipeline.embeddings import get_embedding_model
from rag_pipeline.vector_store import get_or_create_vector_store
from rag_pipeline.retriever import build_retriever


st.set_page_config(page_title="RAG Document Assistant", layout="wide")
st.title("RAG Document Assistant")

DOCUMENTS_DIR = os.getenv("RAG_DOCUMENTS_DIR", "documents")
INDEX_DIR = os.getenv("RAG_INDEX_DIR", os.path.join("rag_pipeline", "faiss_index"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_REPO_ID = os.getenv("RAG_LLM_REPO_ID", "google/flan-t5-base")
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))


@st.cache_resource(show_spinner=True)
def initialize_rag():
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        raise RuntimeError("No documents found in the documents folder.")

    chunks = split_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    embeddings = get_embedding_model(EMBED_MODEL)

    vector_store = get_or_create_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        index_dir=INDEX_DIR,
    )

    retriever = build_retriever(
        vector_store=vector_store,
        k=TOP_K,
        fetch_k=max(12, TOP_K * 3),
    )

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Missing HUGGINGFACEHUB_API_TOKEN. Set it in your terminal before running."
        )

    prompt_template = (
        "You are a concise assistant. Answer only using the provided context.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        "<context>\n"
        "{context}\n"
        "</context>\n\n"
        "Question: {input}\n"
        "Answer:"
    )

    return hf_token, retriever, prompt_template


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def hf_generate(prompt, model_id, token, max_new_tokens=512, temperature=0.2):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "options": {"wait_for_model": True},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data.get("error"))
    if isinstance(data, list) and data:
        if isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    return str(data)


try:
    hf_token, retriever, prompt_template = initialize_rag()
except Exception as exc:
    st.error(f"Initialization failed: {exc}")
    st.stop()

query = st.text_input("Ask a question from notes.txt or research_paper.pdf")

if query:
    start = time.time()
    try:
        docs = retriever.invoke(query)
    except Exception:
        docs = retriever.get_relevant_documents(query)

    context = format_docs(docs)
    prompt = prompt_template.format(context=context, input=query)

    answer = hf_generate(
        prompt=prompt,
        model_id=LLM_REPO_ID,
        token=hf_token,
        max_new_tokens=512,
        temperature=0.2,
    )

    elapsed = time.time() - start

    st.write("### Answer")
    st.write(answer if isinstance(answer, str) else str(answer))

    if docs:
        st.write("### Sources")
        for i, doc in enumerate(docs, start=1):
            source_name = doc.metadata.get("source", "unknown")
            st.write(f"{i}. {source_name}")
    st.caption(f"Response time: {elapsed:.2f}s")
