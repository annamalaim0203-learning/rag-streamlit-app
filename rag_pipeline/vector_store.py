import os
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_or_create_vector_store(chunks, embeddings, index_dir):
    if os.path.isdir(index_dir) and os.listdir(index_dir):
        return FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    return vector_store
