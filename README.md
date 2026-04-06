# RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot for PDF documents.

This project includes:
- `backend/`: FastAPI API for retrieval and chat
- `frontend/`: React + Vite web interface
- `chatbot.py`: terminal chat client using the same RAG pipeline
- `papers/`: PDF files used as the knowledge base

## How It Works

The app:
- loads PDF files from `papers/`
- splits them into chunks
- creates embeddings with `BAAI/bge-small-zh-v1.5`
- stores vectors in FAISS
- retrieves relevant chunks for each question
- asks Ollama `llama3.1:8b` to answer using only retrieved context

## Tech Stack

- Backend: FastAPI, LangChain, FAISS
- Frontend: React, Vite, Axios
- LLM runtime: Ollama
- Embeddings: HuggingFace sentence-transformers

## Requirements

Install these first:
- Python 3.10+
- Node.js 18+
- Ollama

Pull the Ollama model:

```powershell
ollama pull llama3.1:8b
```

If Ollama is running on a different host or port, set these environment variables before starting the backend:

```powershell
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:OLLAMA_MODEL="llama3.1:8b"
```

## Project Structure

```text
RAG_Chatbot/
|-- backend/
|   |-- main.py
|   |-- rag.py
|   `-- requirements.txt
|-- frontend/
|   |-- src/
|   |-- package.json
|   `-- vite.config.js
|-- papers/
|-- chatbot.py
|-- README.md
`-- .gitignore
```

## Run Locally

### 1. Install backend dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### 2. Start the frontend and backend together

```powershell
cd frontend
npm install
npm run dev
```

This dev command now:
- starts the FastAPI backend at `http://127.0.0.1:8000`
- starts the Vite frontend at `http://127.0.0.1:5173`
- warms up the backend by loading PDFs, chunking documents, creating embeddings, and warming the Ollama model
- unlocks the chat UI only after the backend is ready

If you prefer running the backend by itself:

```powershell
uvicorn backend.main:app --reload
```

Available endpoints:
- `GET /health`
- `POST /chat`

Example request body:

```json
{
  "question": "What are the key challenges discussed in the papers?"
}
```

If you want to change the backend URL, create `frontend/.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### 3. Optional: run the terminal chatbot

```powershell
python chatbot.py
```

Type `exit` or `quit` to stop.

## Notes

- Put your PDF files inside `papers/` before asking questions.
- The first run can be slower because the app needs to load documents and build the in-memory vector store.
- If you see `WinError 10061`, the backend could not connect to Ollama. Start the Ollama desktop app or run `ollama serve`, then verify the model is installed with `ollama pull llama3.1:8b`.
- If the backend fails with NumPy-related issues, recreate the virtual environment and reinstall dependencies from [backend/requirements.txt](backend/requirements.txt).
- The web app and terminal chatbot both use the same logic defined in [backend/rag.py](backend/rag.py).
