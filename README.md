# LocalMind AI — Private Knowledge RAG Chatbot

A privacy-first **Retrieval-Augmented Generation (RAG)** chatbot designed for querying private PDF knowledge bases using local Large Language Models (LLMs).

LocalMind AI enables users to interact with their documents through natural language while ensuring that all data processing and inference remain local and secure.

---

## Overview

LocalMind AI implements a complete end-to-end RAG pipeline combining document retrieval, semantic search, and grounded LLM responses.

The system ingests PDF documents, converts them into semantic embeddings, retrieves relevant context at query time, and generates answers strictly based on retrieved knowledge.

Key objectives:

* Local and privacy-preserving document intelligence
* Context-grounded LLM responses
* Modular RAG architecture
* Unified web and terminal interaction interfaces

---

## System Architecture

The application follows a modular client–server design:

```
User Query
    ↓
Frontend (React)
    ↓
FastAPI Backend
    ↓
Document Retriever (FAISS)
    ↓
Context Injection
    ↓
Local LLM (Ollama - Llama 3.1)
    ↓
Grounded Response
```

Core pipeline:

1. Load PDF documents from the knowledge base
2. Split documents into semantic chunks
3. Generate embeddings
4. Store vectors in FAISS
5. Retrieve top-k relevant chunks per query
6. Generate answers using retrieved context only

---

## Features

* Retrieval-Augmented Generation (RAG) pipeline
* Local LLM inference via Ollama
* Semantic search over PDF collections
* Shared backend logic across web and CLI clients
* Automatic backend warm-up and indexing
* Modular architecture for easy experimentation

---

## Tech Stack

### Backend

* FastAPI
* LangChain
* FAISS vector database
* HuggingFace embeddings

### Frontend

* React
* Vite
* Axios

### AI Components

* LLM Runtime: Ollama
* Model: `llama3.1:8b`
* Embeddings: `BAAI/bge-base-en-v1.5`

---

## Project Structure

```
RAG_Chatbot/
│
├── backend/
│   ├── main.py          # FastAPI application
│   ├── rag.py           # RAG pipeline implementation
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── papers/              # Knowledge base (PDF documents)
├── chatbot.py           # Terminal chat client
├── README.md
└── .gitignore
```

---

## Setup

### Prerequisites

* Python 3.10+
* Node.js 18+
* Ollama

Install the LLM model:

```powershell
ollama pull llama3.1:8b
```

---

### Backend Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

---

### Run Development Environment

Start both backend and frontend:

```powershell
cd frontend
npm install
npm run dev
```

Services:

* Backend API → http://127.0.0.1:8000
* Frontend UI → http://127.0.0.1:5173

The backend automatically initializes the RAG pipeline, indexes documents, and prepares the model before enabling chat interaction.

---

### API Endpoints

**Health Check**

```
GET /health
```

**Chat Endpoint**

```
POST /chat
```

Example request:

```json
{
  "question": "What are the key environmental challenges discussed in the papers?"
}
```

---

### Terminal Chat Client (Optional)

```powershell
python chatbot.py
```

---

## Knowledge Base

Place PDF documents inside:

```
papers/
```

Documents are automatically indexed during system initialization.

---

## Design Principles

* **Grounded Generation** — responses rely only on retrieved context
* **Local-First AI** — no external API dependency
* **Modular Components** — easy replacement of models or vector stores
* **Reproducible Pipeline** — shared logic across interfaces

---

## Future Improvements

* Hybrid retrieval (BM25 + embeddings)
* Metadata-aware retrieval
* Streaming responses
* Multi-document citation support
* Evaluation pipeline for RAG quality

---

## License

For research and educational purposes.
