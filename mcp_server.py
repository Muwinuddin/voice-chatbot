"""
mcp_server.py
-------------
Model Context Protocol (MCP) server for the Voice Chatbot.
Built with the FastMCP high-level API (mcp >= 1.0).

MCP lets any MCP-compatible LLM client (Claude Desktop, Cursor, etc.) call
the chatbot's tools directly over stdio or SSE transport.

Tools exposed:
  chat          – send a message and receive an AI response
  search_web    – run a Serper web search
  rag_ingest    – add a text document to the RAG store
  rag_query     – retrieve relevant RAG chunks for a question
  tts           – convert text to speech, returns base64-encoded MP3
  list_sources  – list all ingested RAG documents

Run in stdio mode (for Claude Desktop / Cursor):
  python mcp_server.py

Run with the MCP CLI (stdio):
  mcp run mcp_server.py

Run in SSE mode (HTTP, e.g. for web clients):
  python mcp_server.py --sse --port 8001
"""

import argparse
import base64
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP  # type: ignore[import-untyped]

from groq_client import get_groq_response
from rag_store import RAGStore
from search_client import search_web as _search_web
from tts_client import speak_text
from utils import format_search_results, load_env

load_env()

_rag_store = RAGStore()

# ── FastMCP app ────────────────────────────────────────────────────────────────
mcp = FastMCP("voice-chatbot")


# ── tools ─────────────────────────────────────────────────────────────────────
@mcp.tool()
def chat(
    message: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Send a message to the Voice Chatbot (Groq LLaMA) and get a response.

    Args:
        message: The user message to send.
        history: Optional list of {role, content} dicts for prior turns.
    """
    history = history or []
    history_str = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
        for m in history
    )
    prompt_lines = ["You are a helpful voice assistant. Keep answers concise."]
    if history_str:
        prompt_lines += ["Conversation:", history_str]
    prompt_lines += [f"User: {message}", "Assistant:"]
    prompt = "\n".join(prompt_lines)
    try:
        return get_groq_response(prompt)
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def search_web(query: str, num_results: int = 5) -> str:
    """Search the web using Serper API and return top results.

    Args:
        query: Search query string.
        num_results: Number of results to return (default 5).
    """
    try:
        results = _search_web(query, num_results=num_results)
        return format_search_results(results)
    except Exception as exc:
        return f"Search failed: {exc}"


@mcp.tool()
def rag_ingest(text: str, source_name: str) -> str:
    """Ingest a plain-text document into the RAG knowledge base.

    Args:
        text: Document text to ingest.
        source_name: A label for this document.
    """
    try:
        source_id = _rag_store.ingest_text(text, source_name)
        sources = {s["id"]: s for s in _rag_store.list_sources()}
        chunk_count = sources.get(source_id, {}).get("chunk_count", 0)
        return f"Ingested '{source_name}' as source_id={source_id} ({chunk_count} chunks)."
    except Exception as exc:
        return f"Ingest failed: {exc}"


@mcp.tool()
def rag_query(query: str, n_results: int = 4) -> str:
    """Retrieve the most relevant document chunks from the RAG knowledge base.

    Args:
        query: Question or topic to search for.
        n_results: Number of chunks to return (default 4).
    """
    records = _rag_store.query(query, n_results=n_results)
    return _rag_store.format_context(records) or "No relevant documents found."


@mcp.tool()
def tts(text: str) -> str:
    """Convert text to speech using Azure TTS. Returns base64-encoded MP3.

    Args:
        text: Text to synthesise to speech.
    """
    try:
        audio_bytes = speak_text(text)
        encoded = base64.b64encode(audio_bytes).decode()
        return f"audio/mpeg;base64,{encoded}"
    except Exception as exc:
        return f"TTS failed: {exc}"


@mcp.tool()
def list_sources() -> str:
    """List all documents currently ingested in the RAG knowledge base."""
    sources = _rag_store.list_sources()
    if not sources:
        return "No documents ingested yet."
    lines = [
        f"- [{s['type']}] {s['name']} ({s['chunk_count']} chunks, id={s['id']})"
        for s in sources
    ]
    return "\n".join(lines)


# ── entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Chatbot MCP Server")
    parser.add_argument("--sse", action="store_true", help="Run in SSE/HTTP mode instead of stdio")
    parser.add_argument("--port", type=int, default=8001, help="SSE server port (default 8001)")
    args = parser.parse_args()

    if args.sse:
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
