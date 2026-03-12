"""
rag_store.py
------------
RAG (Retrieval-Augmented Generation) document store built on Chroma + SentenceTransformers.

Responsibilities:
  • Ingest plain text, PDF pages, or arbitrary string chunks into a dedicated
    "documents" Chroma collection (separate from chat memory).
  • Split long texts into overlapping chunks so each chunk fits a language model
    context window.
  • Retrieve the top-k most relevant chunks for a query.
  • Expose a SQLite-backed document registry so the UI can list / delete sources.

Collections used in Chroma:
  rag_documents  — embedded text chunks with source + chunk metadata
"""

import hashlib
import io
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from utils import load_env, normalize_env_value

# ── constants ─────────────────────────────────────────────────────────────────
RAG_COLLECTION = "rag_documents"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 400       # tokens/words (approximate)
DEFAULT_CHUNK_OVERLAP = 80
DB_PATH = Path(__file__).resolve().parent / "rag_sources.db"


# ── SQLite source registry ────────────────────────────────────────────────────
@contextmanager
def _conn():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _bootstrap_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS rag_sources (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                type        TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at  REAL NOT NULL
            )
        """)


_bootstrap_db()


# ── text chunking ─────────────────────────────────────────────────────────────
def _chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── RAG store class ───────────────────────────────────────────────────────────
class RAGStore:
    def __init__(self, embedding_model: str | None = None) -> None:
        load_env()
        model_name = embedding_model or normalize_env_value(os.getenv("EMBEDDING_MODEL", ""))
        if not model_name:
            model_name = DEFAULT_EMBEDDING_MODEL

        self._embedder = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self._client = chromadb.Client(Settings(is_persistent=False))
        self._collection = self._client.get_or_create_collection(
            RAG_COLLECTION,
            embedding_function=self._embedder,
        )

    # ── ingestion ─────────────────────────────────────────────────────────────

    def ingest_text(
        self,
        text: str,
        source_name: str,
        source_type: str = "text",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> str:
        """Chunk and embed raw text. Returns source_id."""
        source_id = hashlib.sha256(f"{source_name}:{time.time()}".encode()).hexdigest()[:16]
        chunks = _chunk_text(text, chunk_size, overlap)
        if not chunks:
            return source_id

        ids, documents, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid4()))
            documents.append(chunk)
            metadatas.append({"source_id": source_id, "source_name": source_name, "chunk_index": i})

        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

        with _conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO rag_sources (id, name, type, chunk_count, created_at) VALUES (?, ?, ?, ?, ?)",
                (source_id, source_name, source_type, len(chunks), time.time()),
            )
        return source_id

    def ingest_pdf_bytes(self, pdf_bytes: bytes, source_name: str) -> str:
        """Extract text from PDF bytes and ingest. Requires pypdf."""
        try:
            import pypdf  # type: ignore
        except ImportError as exc:
            raise ImportError("Install pypdf to ingest PDFs: pip install pypdf") from exc

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages)
        return self.ingest_text(full_text, source_name, source_type="pdf")

    # ── retrieval ─────────────────────────────────────────────────────────────

    def query(self, query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Return top-k relevant chunks for a query."""
        if not query_text:
            return []
        try:
            result = self._collection.query(query_texts=[query_text], n_results=n_results)
        except Exception:
            return []

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        records = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            records.append({
                "chunk": doc,
                "source_name": (meta or {}).get("source_name", ""),
                "chunk_index": (meta or {}).get("chunk_index", 0),
                "score": round(1 - dist, 4),
            })
        return records

    def format_context(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return ""
        lines = ["Retrieved document context:"]
        for r in records:
            lines.append(f"[{r['source_name']}] {r['chunk']}")
        return "\n".join(lines)

    # ── management ────────────────────────────────────────────────────────────

    def list_sources(self) -> list[dict]:
        with _conn() as con:
            rows = con.execute(
                "SELECT id, name, type, chunk_count, created_at FROM rag_sources ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_source(self, source_id: str) -> int:
        """Delete all chunks for a source and remove from registry. Returns chunk count deleted."""
        result = self._collection.get(where={"source_id": {"$eq": source_id}})
        ids_to_delete = result.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        with _conn() as con:
            con.execute("DELETE FROM rag_sources WHERE id = ?", (source_id,))
        return len(ids_to_delete)
