import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from groq_client import get_groq_response
from rag_store import RAGStore
from search_client import search_web
from tts_client import speak_text
from utils import (
    extract_profile_facts,
    format_memory_snippets,
    format_search_results,
    load_env,
    normalize_env_value,
)
from vector_store import VectorMemory

load_env()

app = FastAPI(title="Voice Chatbot API")
memory_store = VectorMemory()
rag_store = RAGStore()

cors_origins = normalize_env_value(os.getenv("BACKEND_CORS_ORIGINS", ""))
if cors_origins:
    origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] | None = None
    use_search: bool = True


class ChatResponse(BaseModel):
    response: str
    search_results: list[dict]


class TTSRequest(BaseModel):
    text: str


class IngestTextRequest(BaseModel):
    text: str
    source_name: str


class IngestResponse(BaseModel):
    source_id: str
    chunk_count: int
    source_name: str


class RAGSource(BaseModel):
    id: str
    name: str
    type: str
    chunk_count: int
    created_at: float


def should_search(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    keywords = [
        "search",
        "web",
        "internet",
        "latest",
        "news",
        "current",
        "today",
        "recent",
        "lookup",
        "find",
        "price",
        "stock",
        "weather",
        "score",
    ]
    if any(word in t for word in keywords):
        return True
    question_words = ["who", "what", "when", "where", "why", "how"]
    return "?" in t and any(word in t for word in question_words)


def build_prompt(
    history: list[ChatMessage],
    search_context: str,
    memory_context: str,
    rag_context: str = "",
) -> str:
    lines = [
        "You are a helpful voice assistant. Keep answers clear and concise.",
    ]
    if memory_context:
        lines.append("Relevant memory:")
        lines.append(memory_context)
    if rag_context:
        lines.append(rag_context)
    if search_context:
        lines.append("Use these web search results when relevant:")
        lines.append(search_context)
    lines.append("Conversation:")
    for msg in history:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def normalize_history(history: list[ChatMessage] | None, message: str) -> list[ChatMessage]:
    if history:
        return history
    return [ChatMessage(role="user", content=message)]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    history = normalize_history(request.history, request.message)
    search_results: list[dict[str, Any]] = []
    search_context = ""

    memory_records = memory_store.query(request.message, n_results=4, types=["chat", "profile"])
    memory_context = format_memory_snippets(memory_records)

    rag_records = rag_store.query(request.message, n_results=4)
    rag_context = rag_store.format_context(rag_records)

    if request.use_search and should_search(request.message):
        try:
            search_results = search_web(request.message)
            search_context = format_search_results(search_results)
        except Exception:
            search_results = []
            search_context = ""

    try:
        prompt = build_prompt(history, search_context, memory_context, rag_context)
        response_text = get_groq_response(prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    memory_store.add_text(request.message, role="user", mem_type="chat")
    for fact in extract_profile_facts(request.message):
        memory_store.add_text(fact, role="user", mem_type="profile")
    memory_store.add_text(response_text, role="assistant", mem_type="chat")

    return ChatResponse(response=response_text, search_results=search_results)


@app.post("/tts")
def tts(request: TTSRequest) -> Response:
    try:
        audio_bytes = speak_text(request.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ── RAG endpoints ─────────────────────────────────────────────────────────────

@app.post("/rag/ingest/text", response_model=IngestResponse)
def rag_ingest_text(request: IngestTextRequest) -> IngestResponse:
    """Ingest a plain-text document into the RAG store."""
    try:
        source_id = rag_store.ingest_text(request.text, request.source_name)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    sources = {s["id"]: s for s in rag_store.list_sources()}
    chunk_count = sources.get(source_id, {}).get("chunk_count", 0)
    return IngestResponse(source_id=source_id, chunk_count=chunk_count, source_name=request.source_name)


@app.post("/rag/ingest/pdf", response_model=IngestResponse)
async def rag_ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    """Upload and ingest a PDF file into the RAG store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    pdf_bytes = await file.read()
    try:
        source_id = rag_store.ingest_pdf_bytes(pdf_bytes, file.filename)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    sources = {s["id"]: s for s in rag_store.list_sources()}
    chunk_count = sources.get(source_id, {}).get("chunk_count", 0)
    return IngestResponse(source_id=source_id, chunk_count=chunk_count, source_name=file.filename)


@app.get("/rag/sources", response_model=list[RAGSource])
def rag_list_sources() -> list[RAGSource]:
    return [RAGSource(**s) for s in rag_store.list_sources()]


@app.delete("/rag/sources/{source_id}")
def rag_delete_source(source_id: str) -> dict:
    deleted = rag_store.delete_source(source_id)
    return {"deleted_chunks": deleted}

    return Response(content=audio_bytes, media_type="audio/mpeg")
