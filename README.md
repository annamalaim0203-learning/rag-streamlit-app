# RAG Document Assistant (Streamlit)

A Retrieval‑Augmented Generation (RAG) app built with Streamlit that loads local documents, indexes them into a vector database, retrieves relevant chunks, and generates grounded answers using an LLM.

---

## What Is RAG?

**RAG (Retrieval‑Augmented Generation)** combines:
1. **Retrieval** — find relevant chunks from your data  
2. **Generation** — answer using an LLM grounded on retrieved context  

This improves accuracy and reduces hallucinations by anchoring responses in real documents.

---

## Terms and Definitions

- **Document Loader**: Reads files (PDF/TXT) and returns structured document objects.
- **Chunking / Splitter**: Breaks documents into smaller overlapping pieces for efficient search.
- **Embedding**: Vector representation of text used for similarity search.
- **Vector Store**: Database (FAISS) that stores embeddings and supports fast nearest‑neighbor search.
- **Retriever**: Component that returns the most relevant chunks for a query.
- **MMR (Maximal Marginal Relevance)**: Retrieval strategy that balances relevance + diversity.
- **LLM (Large Language Model)**: Generates the final response from the retrieved context.
- **Prompt**: Template used to instruct the model to answer only from context.
- **Grounding**: Ensuring answers are based strictly on retrieved sources.
- **Index Persistence**: Saving the vector store to disk for reuse across sessions.

---

## Workflow (High‑Level)

1. Load documents from `documents/`  
2. Split into overlapping chunks  
3. Embed chunks with a sentence‑transformer model  
4. Persist embeddings in FAISS  
5. Retrieve top‑K relevant chunks  
6. Generate answer using retrieved context  
7. Display answer + sources  

---

## Simple Analogy

RAG is like an open‑book exam:
- Retrieval = finding the right pages  
- LLM = writing the answer  
- Prompt = the question paper  

---

## Hugging Face Token (Step‑By‑Step)

1. Go to https://huggingface.co and log in  
2. Click your profile → **Settings**  
3. Click **Access Tokens**  
4. Click **New token**  
5. Name it (e.g., `rag-streamlit-app`)  
6. Choose permission:
   - **Inference → Make calls to Inference Providers**
7. Generate and copy the token  

Set it in PowerShell:
```powershell
$env:HUGGINGFACEHUB_API_TOKEN="YOUR_HF_TOKEN"
```

---

## Errors Encountered and Mitigation

### 1) `ModuleNotFoundError: No module named 'langchain.chains'`
**Cause:** Installed LangChain version did not include `langchain.chains`  
**Mitigation:** Removed `langchain.chains` usage and used a direct prompt + LLM call.

### 2) `ImportError: cannot import name 'get_or_create_vector_store'`
**Cause:** Function was missing in `vector_store.py`  
**Mitigation:** Added `get_or_create_vector_store()` and updated imports.

### 3) `split_documents() got an unexpected keyword argument 'chunk_size'`
**Cause:** Function signature did not accept parameters  
**Mitigation:** Added `chunk_size` and `chunk_overlap` parameters.

### 4) Same errors persisted after fixes
**Cause:** Running a different project folder (`Desktop` vs `D:\projects`)  
**Mitigation:** Standardized on `D:\projects\rag-streamlit-app` and synced files.

### 5) `get_embedding_model() takes 0 positional arguments but 1 was given`
**Cause:** `get_embedding_model()` did not accept a model name argument  
**Mitigation:** Updated function to accept `model_name` with a default.

### 6) `Invalid task None` / Hugging Face Hub validation errors
**Cause:** Token/task not passed into the LLM call  
**Mitigation:** Added explicit token handling and clear error messages.

### 7) `AttributeError: 'InferenceClient' object has no attribute 'post'`
**Cause:** Incompatible `HuggingFaceHub` wrapper with current `huggingface_hub` client  
**Mitigation:** Removed LangChain’s HF wrapper and used direct inference calls.

### 8) `StopIteration` in HF provider resolution
**Cause:** Provider not specified when using `InferenceClient`  
**Mitigation:** Forced explicit provider or removed client wrapper.

### 9) `AttributeError: 'InferenceClient' object has no attribute 'text2text_generation'`
**Cause:** Client version mismatch with API method names  
**Mitigation:** Switched to direct REST call using `requests`.

### 10) `HTTPError: 410 Gone` for `google/flan-t5-base`
**Cause:** Model no longer available via public Inference API at that endpoint  
**Mitigation:** Change to a currently supported model or host the model yourself.

---

## Error Messages (Full List Captured)

- `ModuleNotFoundError: No module named 'langchain.chains'`
- `ImportError: cannot import name 'get_or_create_vector_store'`
- `split_documents() got an unexpected keyword argument 'chunk_size'`
- `get_embedding_model() takes 0 positional arguments but 1 was given`
- `Invalid task None` / HF Hub validation error
- `AttributeError: 'InferenceClient' object has no attribute 'post'`
- `StopIteration` in provider selection
- `AttributeError: 'InferenceClient' object has no attribute 'text2text_generation'`
- `requests.exceptions.HTTPError: 410 Client Error: Gone`

---

## Mitigation Plan (Summary)

- Use a stable folder (`D:\projects\rag-streamlit-app`) and keep it as source of truth  
- Avoid unsupported LangChain wrappers for Hugging Face  
- Use direct REST calls to HF Inference API  
- Provide explicit token checks with clear failure messages  
- Swap to a supported model if public inference returns 410  

---

## Project Structure

```
rag-streamlit-app/
├── app.py
├── requirements.txt
├── documents/
├── rag_pipeline/
│   ├── loader.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── retriever.py
```

---

## Notes

- This README reflects all steps, errors, and fixes performed during the build process.
- The final model choice must be supported by the Hugging Face Inference API.
