import io
import logging
import os
import re
import socket
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "papers"
TOP_K = 5
RELEVANCE_THRESHOLD = 0.2
FETCH_K = 20
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
MIN_KEYWORD_LENGTH = 4
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "paper",
    "the",
    "this",
    "to",
    "what",
    "with",
}

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

for logger_name in ("sentence_transformers", "transformers", "huggingface_hub", "unstructured"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

MARKDOWN_SEPARATORS = [
    # Regex separators tuned for the three environmental-pollution PDFs:
    # - clean numbered sections in the ChatGPT paper
    # - single-line headings with repeated spaces in the Gemini paper
    # - ALL-CAPS journal headings in the original paper
    r"\n(?=#{1,6}\s)",
    r"\n(?=[A-Z][A-Z][A-Z \-:&()]{3,}\n)",
    r"\n(?=(?:Abstract|Keywords|Introduction|Methodology|Methods?|Materials(?: and Methods?)?|Results(?: and Discussion)?|Discussion|Conclusion|References)\b)",
    r"\n(?=\d+(?:\.\d+)?\.\s{1,}(?:Abstract|Keywords|Introduction|Methodology|Methods?|Current Dimensions of Pollution|Results(?: and Discussion)?|Discussion|Proposed Solutions(?: and Strategic Framework)?|Conclusion|References(?:\s*\(Simplified\))?)\b)",
    r"\n(?=\d+\.\d+(?:\.\d+)?\.\s{1,}[A-Z][A-Za-z\"()/-]*(?:\s{1,}(?:[A-Z][A-Za-z\"()/-]*|of|and|the|to|in|for|on|with|&)){0,12}(?:\s{2,}|\n))",
    r"\n(?=(?:Table|TABLE|Figure|FIGURE|Fig\.)\s*\d*)",
    r"\n(?=Strategy\s+Level\s+Action\s+Items\s+Expected\s+Impact\b)",
    "\n(?=[\\-*\\u2022\\u25A0\\u25CF\\x7f])",
    r"\n{3,}",
    r"\n{2,}",
    r"\n",
    r"(?<=[.?!:;])\s+",
    r" {2,}",
    r" ",
]

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    (
        "You are a helpful assistant for a private knowledge base.\n"
        "RULES:\n"
        "1. Use ONLY the provided context to answer.\n"
        "2. If the answer is not clearly contained in the context, say: "
        "\"I don't know based on the provided documents.\"\n"
        "3. Do NOT use outside knowledge, guessing, or web information.\n"
        "4. Answer in a natural, smooth, easy-to-read way.\n"
        "5. Do NOT cite sources after each sentence.\n"
        "6. If you use the context, add a final line at the end in this format only: "
        "\"Sources: source1, source2\".\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n"
    )
)


@contextmanager
def _quiet_console():
    """Silence noisy third-party stdout/stderr and non-critical warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield


def _create_loader() -> DirectoryLoader:
    return DirectoryLoader(
        path=str(PAPERS_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=False,
        use_multithreading=True,
    )


def _create_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
        is_separator_regex=True,
        separators=MARKDOWN_SEPARATORS,
    )


def _create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _create_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )


def _get_ollama_host_port() -> tuple[str, int]:
    parsed = urlparse(OLLAMA_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is not None:
        return host, parsed.port
    return host, 443 if parsed.scheme == "https" else 11434


def _format_ollama_error(exc: Exception | None = None) -> str:
    host, port = _get_ollama_host_port()
    message = (
        f"Cannot reach Ollama at {OLLAMA_BASE_URL} ({host}:{port}). "
        "Start the Ollama app or run `ollama serve`, then make sure the model "
        f"`{OLLAMA_MODEL}` is installed with `ollama pull {OLLAMA_MODEL}`."
    )
    if exc is None:
        return message
    return f"{message} Original error: {exc}"


def _ensure_ollama_available() -> None:
    host, port = _get_ollama_host_port()
    try:
        with socket.create_connection((host, port), timeout=2):
            return
    except OSError as exc:
        raise RuntimeError(_format_ollama_error(exc)) from exc


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    _ensure_ollama_available()
    return _create_llm()


def _format_source(metadata: dict) -> str:
    source = metadata.get("source", "Unknown source")
    source_name = Path(source).name
    page = metadata.get("page")
    if isinstance(page, int):
        return f"{source_name}:{page + 1}"
    return source_name


def _format_context(docs: list) -> str:
    chunks = []
    for doc in docs:
        source = _format_source(doc.metadata)
        chunks.append(f"[{source}]\n{doc.page_content}")
    return "\n\n".join(chunks)


def _extract_sources(docs: list) -> list[str]:
    seen = set()
    sources = []
    for doc in docs:
        source = _format_source(doc.metadata)
        if source in seen:
            continue
        seen.add(source)
        sources.append(source)
    return sources


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", text.lower())).strip()


def _extract_keywords(question: str) -> list[str]:
    normalized_question = _normalize_text(question)
    keywords = []
    for token in normalized_question.split():
        if len(token) < MIN_KEYWORD_LENGTH or token in STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
    return keywords


def _extract_phrases(question: str) -> list[str]:
    tokens = _normalize_text(question).split()
    phrases = []
    for size in range(2, 5):
        for index in range(len(tokens) - size + 1):
            phrase_tokens = tokens[index : index + size]
            if sum(token not in STOPWORDS for token in phrase_tokens) < 2:
                continue
            phrase = " ".join(phrase_tokens)
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases


def _section_phrase_search(question: str, k: int = TOP_K) -> list:
    candidate_phrases = []
    for phrase in _extract_phrases(question):
        tokens = phrase.split()
        if "paper" in tokens:
            continue
        if sum(token not in STOPWORDS for token in tokens) < 2:
            continue
        candidate_phrases.append(phrase)

    chunks = get_chunks()
    phrase_frequencies = {}
    for phrase in candidate_phrases:
        phrase_frequencies[phrase] = sum(
            1 for doc in chunks if phrase in _normalize_text(doc.page_content)
        )

    scored_docs = []
    for doc in chunks:
        normalized_doc = _normalize_text(doc.page_content)
        matched_phrases = [phrase for phrase in candidate_phrases if phrase in normalized_doc]
        if not matched_phrases:
            continue

        score = 0.0
        for phrase in matched_phrases:
            content_tokens = len([token for token in phrase.split() if token not in STOPWORDS])
            frequency = max(phrase_frequencies.get(phrase, 1), 1)
            score += content_tokens / frequency
        scored_docs.append((score, doc))

    scored_docs.sort(
        key=lambda item: (
            -item[0],
            Path(item[1].metadata.get("source", "")).name,
            item[1].metadata.get("page", 0),
            item[1].metadata.get("start_index", 0),
        )
    )
    if not scored_docs:
        return []

    best_score = scored_docs[0][0]
    kept_docs = [doc for score, doc in scored_docs if score >= best_score * 0.75]
    return kept_docs[:k]


def _keyword_score(question: str, doc_text: str, metadata: dict | None = None) -> float:
    normalized_question = _normalize_text(question)
    normalized_doc = _normalize_text(doc_text)
    if not normalized_question or not normalized_doc:
        return 0.0

    score = 0.0
    if normalized_question in normalized_doc:
        score += 3.0

    keywords = _extract_keywords(question)
    for keyword in keywords:
        if keyword in normalized_doc:
            score += 1.0

    for phrase in _extract_phrases(question):
        if phrase in normalized_doc:
            score += 2.0

    if len(keywords) >= 2:
        for left, right in zip(keywords, keywords[1:]):
            phrase = f"{left} {right}"
            if phrase in normalized_doc:
                score += 0.5

    if metadata:
        source = Path(metadata.get("source", "")).stem
        normalized_source = _normalize_text(source)
        for keyword in keywords:
            if keyword in normalized_source:
                score += 0.75

        if "paper" in normalized_question and "paper" in normalized_source:
            score += 2.0

    return score


def _append_sources(answer: str, sources: list[str]) -> str:
    cleaned_answer = answer.strip()
    if not sources:
        return cleaned_answer

    lower_answer = cleaned_answer.lower()
    if "sources:" in lower_answer:
        head, _, _ = cleaned_answer.rpartition("Sources:")
        cleaned_answer = head.strip()

    return f"{cleaned_answer}\n\nSources: {', '.join(sources)}"


@lru_cache(maxsize=1)
def get_chunks():
    if not PAPERS_DIR.exists():
        raise FileNotFoundError(f"Missing papers directory: {PAPERS_DIR}")

    with _quiet_console():
        docs = _create_loader().load()
    if not docs:
        raise ValueError(f"No PDF files found in {PAPERS_DIR}")

    return _create_text_splitter().split_documents(docs)


def _keyword_fallback_search(question: str, k: int = TOP_K) -> list:
    scored_docs = []
    for doc in get_chunks():
        score = _keyword_score(question, doc.page_content, doc.metadata)
        if score <= 0:
            continue
        scored_docs.append((score, doc))

    scored_docs.sort(
        key=lambda item: (
            -item[0],
            Path(item[1].metadata.get("source", "")).name,
            item[1].metadata.get("page", 0),
            item[1].metadata.get("start_index", 0),
        )
    )
    return [doc for _score, doc in scored_docs[:k]]


@lru_cache(maxsize=1)
def get_vectorstore():
    splits = get_chunks()
    with _quiet_console():
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=_create_embeddings(),
            distance_strategy=DistanceStrategy.COSINE,
        )
    return vectorstore


def ask_rag(question: str) -> dict:
    cleaned_question = question.strip()
    if not cleaned_question:
        return {"answer": "Please enter a question.", "sources": []}

    vectorstore = get_vectorstore()
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        cleaned_question,
        k=TOP_K,
        fetch_k=FETCH_K,
        score_threshold=RELEVANCE_THRESHOLD,
    )
    docs = [doc for doc, _score in docs_with_scores]
    if not docs:
        docs = _section_phrase_search(cleaned_question, k=TOP_K)
    if not docs:
        docs = _keyword_fallback_search(cleaned_question, k=TOP_K)
    sources = _extract_sources(docs)

    if not docs:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
        }

    prompt_value = PROMPT_TEMPLATE.invoke(
        {"context": _format_context(docs), "question": cleaned_question}
    )
    try:
        response = get_llm().invoke(prompt_value)
    except Exception as exc:
        raise RuntimeError(_format_ollama_error(exc)) from exc
    answer = response.content if hasattr(response, "content") else str(response)
    answer = _append_sources(answer, sources)

    return {"answer": answer, "sources": sources}


def warm_up() -> None:
    """Prepare the retriever and warm the local LLM before first user query."""
    get_vectorstore()
    _ensure_ollama_available()
    with _quiet_console():
        try:
            get_llm().invoke("Reply with the single word READY.")
        except Exception as exc:
            raise RuntimeError(_format_ollama_error(exc)) from exc
