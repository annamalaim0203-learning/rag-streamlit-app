# 📚 Retrieval-Augmented Generation (RAG) – Streamlit AI Application

## Overview

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** based AI application using **Python, Streamlit, LangChain, FAISS, and Embedding Models**.

The goal of this project is **R&D and learning**:  
To understand how modern AI systems combine **information retrieval + large language models** to answer questions based on custom documents.

The application allows a user to:

1. Store documents
2. Convert them into embeddings
3. Store them in a vector database
4. Retrieve relevant information
5. Generate answers using an LLM

---

# What is RAG (Retrieval-Augmented Generation)

### Definition
Retrieval-Augmented Generation is an AI architecture where an LLM retrieves relevant information from external documents before generating an answer.

Instead of relying only on training data, the model uses **real-time knowledge retrieval**.

### Simple Meaning
**RAG = Search Engine + AI Writer**

The system first **searches for relevant information**, then the **AI generates the answer using that information**.

---

# Why RAG is Important

Large Language Models like GPT have limitations:

- Knowledge cutoff
- Hallucination issues
- Cannot access private data

RAG solves these problems by allowing AI models to use **external knowledge sources**.

---

# Core Components of a RAG System

## 1. Documents

Documents are the **knowledge source**.

Examples:

- PDFs
- Text files
- Research papers
- Knowledge bases
- Websites

Example project folder:

```
documents/
    notes.txt
    research_paper.pdf
```

---

# 2. Document Loader

### Definition
A document loader reads files and converts them into a format that AI systems can process.

Examples:

```
TextLoader
PyPDFLoader
```

### Analogy
Think of a loader like a **scanner** that reads books and converts them into digital text.

---

# 3. Text Chunking

### Definition
Chunking splits large documents into smaller pieces.

Why?

LLMs have **token limits** and cannot process entire documents at once.

Example:

```
chunk_size = 500
chunk_overlap = 50
```

### Analogy
Imagine reading a **huge textbook**.  
Instead of reading the whole book, you break it into **small paragraphs**.

---

# 4. Embeddings

### Definition
Embeddings convert text into **numerical vectors** that represent meaning.

Example:

```
AI → [0.23, 0.91, 0.12, ...]
Machine Learning → [0.21, 0.88, 0.15, ...]
```

Texts with similar meaning will have **similar vectors**.

### Analogy
Embedding is like converting words into **GPS coordinates on a map**.

Words with similar meaning appear **closer together on the map**.

---

# 5. Vector Database

### Definition
A vector database stores embeddings and allows similarity search.

Popular vector databases:

- FAISS
- Chroma
- Pinecone
- Weaviate

In this project we used:

```
FAISS
```

### Analogy
A vector database works like a **library where books are organized by meaning instead of alphabet**.

---

1. FAISS

Definition (one line):
FAISS is a library used to store embeddings (vectors) and quickly find the most similar ones.

Simple Analogy:
FAISS is like a super-fast librarian who finds books with similar meaning instead of similar titles. 📚

---

# 6. Retriever

### Definition
The retriever finds the most relevant document chunks based on the user query.

Steps:

1. Convert query into embedding
2. Compare with stored embeddings
3. Retrieve top matching documents

### Analogy
Like asking a librarian:

> "Find the most relevant books about this topic."

---

# 7. Large Language Model (LLM)

### Definition
A Large Language Model generates human-like text.

Examples:

- GPT
- Gemini
- LLaMA
- Flan-T5

In this project we used:

```
google/flan-t5-base
```

### Analogy
Think of the LLM as an **AI writer who reads retrieved documents and writes a final answer**.

---

2. FLAN-T5

Definition (one line):
FLAN-T5 is a pretrained language model that understands instructions and generates answers in natural language.

Simple Analogy:
FLAN-T5 is like a smart student who reads information and writes a clear answer for your question. 🧠

3. LangChain

Definition (one line):
LangChain is a framework that connects LLMs, documents, embeddings, and vector databases to build AI applications.

Simple Analogy:
LangChain is like a manager who coordinates the search engine, documents, and AI model to produce the final answer. 🧩

---

# RAG Workflow

Complete workflow:

```
User Question
      ↓
Convert question to embedding
      ↓
Search vector database
      ↓
Retrieve relevant document chunks
      ↓
Provide context to LLM
      ↓
LLM generates final answer
```

Simplified architecture:

```
Documents
   ↓
Loader
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Database
   ↓
Retriever
   ↓
LLM
   ↓
Answer
```

---

# Application Architecture

```
rag-streamlit-app
│
├── app.py
├── requirements.txt
│
├── documents
│   ├── notes.txt
│   └── research_paper.pdf
│
└── rag_pipeline
    ├── loader.py
    ├── embeddings.py
    ├── vector_store.py
    └── retriever.py
```

---

# Technology Stack

| Component | Tool |
|--------|------|
| UI | Streamlit |
| Language | Python |
| LLM Framework | LangChain |
| Embeddings | Sentence Transformers |
| Vector DB | FAISS |
| Model | Flan-T5 |

---

# Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Application

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

# Challenges Faced During Development

## 1. LangChain Version Issues

### Error

```
ModuleNotFoundError: No module named 'langchain.chains'
```

### Reason
New versions of LangChain removed or reorganized modules. All these issues are mostly due to version constraints of python and computational resource constraints.

### Fix
Avoid deprecated imports like:

```
from langchain.chains import RetrievalQA
```

Use simplified retrieval logic.

---

## 2. Text Splitter Import Error

### Error

```
ModuleNotFoundError: langchain.text_splitter
```

### Reason
LangChain moved text splitters to a separate package. All these issues are mostly due to version constraints of python and computational resource constraints.

### Fix

Install:

```
pip install langchain-text-splitters
```

Import:

```
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

---

## 3. Indentation Errors

### Error

```
IndentationError: expected an indented block
```

### Reason
Python requires indentation after function definitions. All these issues are mostly due to version constraints of python and computational resource constraints.

Incorrect:

```
def function():
print("Hello")
```

Correct:

```
def function():
    print("Hello")
```

---

## 4. Dependency Conflicts

LangChain ecosystem was split into multiple libraries:

```
langchain
langchain-core
langchain-community
langchain-text-splitters
```

### Fix

Use compatible dependencies in `requirements.txt`.

---

# Lessons Learned

- RAG systems require **multiple components working together**
- Dependency management is critical in AI frameworks
- Vector databases enable **semantic search**
- Chunking improves retrieval performance
- Proper project structure simplifies debugging

---

# Future Improvements

Possible upgrades:

- Document upload feature
- Persistent vector database
- Conversational memory
- Hybrid search (keyword + vector)
- Multiple LLM support
- Production deployment

---

# Conclusion

This project demonstrates the **complete lifecycle of a RAG application**, including:

- Document ingestion
- Semantic embedding
- Vector search
- Retrieval
- LLM-based answer generation
- Streamlit interface

Understanding RAG is essential for modern AI systems because it enables **LLMs to interact with real-world knowledge sources**.

Commands to execute the model:

cd D:\projects\rag-streamlit-app
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:HUGGINGFACEHUB_API_TOKEN="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"
streamlit run app.py

---

Absolutely—here are the token steps in simple pointers:

1. Go to https://huggingface.co and log in  
2. Click your profile icon → **Settings**  
3. Click **Access Tokens**  
4. Click **New token**  
5. Enter a name (e.g., `rag-streamlit-app`)  
6. Select permission: **Inference → Make calls to Inference Providers**  
7. Click **Generate** and copy the token  

Then in PowerShell:
```powershell
$env:HUGGINGFACEHUB_API_TOKEN="YOUR_HF_TOKEN"
```

# Author

AI / ML Engineering Learning Project
