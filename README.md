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

### 1. Start the backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`.

Available endpoints:
- `GET /health`
- `POST /chat`

Example request body:

```json
{
  "question": "What are the key challenges discussed in the papers?"
}
```

### 2. Start the frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend runs at `http://127.0.0.1:5173`.

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
- If the backend fails with NumPy-related issues, recreate the virtual environment and reinstall from [backend/requirements.txt](d:/Courses/LLM/RAG_Chatbot/backend/requirements.txt).
- The web app and terminal chatbot both use the same logic from [backend/rag.py](d:/Courses/LLM/RAG_Chatbot/backend/rag.py).

## Push To GitHub

Recommended steps:

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

Before pushing, make sure these are not included:
- `.venv/`
- `frontend/node_modules/`
- `frontend/dist/`
- local `.env` files

## Deployment Note

You can push this repo to GitHub without problems.

For hosting:
- the frontend can be deployed to services like Vercel or Netlify
- the backend needs a Python host
- Ollama usually needs to run on your own machine or a separate server

So GitHub is a good place to store the source code, but not enough by itself to run the full RAG app online.
