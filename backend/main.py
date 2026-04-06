from contextlib import asynccontextmanager
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


def _run_warmup(app: FastAPI) -> None:
    from backend.rag import warm_up

    try:
        app.state.backend_status = "loading"
        app.state.backend_error = ""
        warm_up()
        app.state.backend_status = "ready"
    except Exception as exc:
        app.state.backend_status = "error"
        app.state.backend_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.backend_status = "loading"
    app.state.backend_error = ""
    Thread(target=_run_warmup, args=(app,), daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)

# Allow React to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {
        "status": app.state.backend_status,
        "ready": app.state.backend_status == "ready",
        "error": app.state.backend_error,
    }


@app.post("/chat")
def chat(data: Question) -> ChatResponse:
    if app.state.backend_status == "loading":
        raise HTTPException(status_code=503, detail="Backend is still preparing documents and models.")
    if app.state.backend_status == "error":
        raise HTTPException(status_code=500, detail=app.state.backend_error or "Backend warm-up failed.")

    try:
        from backend.rag import ask_rag

        result = ask_rag(data.question)
        return ChatResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
