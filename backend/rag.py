from functools import lru_cache
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "papers"

MARKDOWN_SEPARATORS = [
    "\nAbstract\n",
    "\nIntroduction\n",
    "\nMitigation Policy Instruments\n",
    "\nEconomic and Social Impacts\n",
    "\nChallenges and Limitations\n",
    "\nFuture Directions\n",
    "\nConclusion\n",
    "\nGreen Infrastructure Concepts\n",
    "\nEcosystem Services\n",
    "\nPlanning and Governance\n",
    "\nChallenges\n",
    "\n\n",
    ". ",
    "? ",
    "! ",
    "\n",
    " ",
    "",
]

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    (
        "You are a strict, citation-focused assistant for a private knowledge base.\n"
        "RULES:\n"
        "1. Use ONLY the provided context to answer.\n"
        "2. If the answer is not clearly contained in the context, say: "
        "\"I don't know based on the provided documents.\"\n"
        "3. Do NOT use outside knowledge, guessing, or web information.\n"
        "4. If applicable, cite sources as (source:page) using the metadata.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n"
    )
)


def _create_loader() -> DirectoryLoader:
    return DirectoryLoader(
        path=str(PAPERS_DIR),
        glob="**/*.pdf",
        loader_cls=UnstructuredFileLoader,
        show_progress=True,
        use_multithreading=True,
    )


def _create_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )


def _create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


def _create_llm() -> ChatOllama:
    return ChatOllama(model="llama3.1:8b", temperature=0.0)


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


@lru_cache(maxsize=1)
def get_retriever():
    if not PAPERS_DIR.exists():
        raise FileNotFoundError(f"Missing papers directory: {PAPERS_DIR}")

    docs = _create_loader().load()
    if not docs:
        raise ValueError(f"No PDF files found in {PAPERS_DIR}")

    splits = _create_text_splitter().split_documents(docs)
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=_create_embeddings(),
        distance_strategy=DistanceStrategy.COSINE,
    )
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.2},
    )


def ask_rag(question: str) -> dict:
    cleaned_question = question.strip()
    if not cleaned_question:
        return {"answer": "Please enter a question.", "sources": []}

    retriever = get_retriever()
    docs = retriever.invoke(cleaned_question)
    sources = _extract_sources(docs)

    if not docs:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
        }

    prompt_value = PROMPT_TEMPLATE.invoke(
        {"context": _format_context(docs), "question": cleaned_question}
    )
    response = _create_llm().invoke(prompt_value)
    answer = response.content if hasattr(response, "content") else str(response)

    return {"answer": answer, "sources": sources}
