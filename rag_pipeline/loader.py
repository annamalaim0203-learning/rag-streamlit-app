import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents(folder_path="documents"):
    documents = []
    if not os.path.isdir(folder_path):
        return documents

    for file in os.listdir(folder_path):
        if file.startswith("."):
            continue

        path = os.path.join(folder_path, file)

        try:
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
                documents.extend(loader.load())
            elif file.lower().endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
                documents.extend(loader.load())
        except Exception:
            continue

    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split_text(text):
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            if end == length:
                break
            start = max(0, end - chunk_overlap)
        return chunks

    split_docs = []
    for doc in documents:
        for chunk in split_text(doc.page_content):
            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return split_docs
