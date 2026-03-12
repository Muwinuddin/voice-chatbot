"""
history_store.py
----------------
SQLite-backed persistent chat session and message store.
Built on Python stdlib sqlite3 — no extra dependencies.

Schema
  sessions : id TEXT PK | title TEXT | created_at REAL | updated_at REAL
  messages : id INTEGER PK | session_id TEXT FK | role TEXT | content TEXT | ts REAL
"""

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "chat_history.db"


@contextmanager
def _conn():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _bootstrap() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT    PRIMARY KEY,
                title      TEXT    NOT NULL,
                created_at REAL    NOT NULL,
                updated_at REAL    NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                ts         REAL    NOT NULL
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, ts)")


_bootstrap()


def create_session(title: str = "New chat") -> str:
    """Create a new chat session and return its ID."""
    sid = str(uuid.uuid4())
    now = time.time()
    with _conn() as con:
        con.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (sid, title, now, now),
        )
    return sid


def rename_session(session_id: str, title: str) -> None:
    """Rename a session (e.g. after the first user message)."""
    with _conn() as con:
        con.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title[:80], time.time(), session_id),
        )


def delete_session(session_id: str) -> None:
    """Delete a session and all its messages."""
    with _conn() as con:
        con.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        con.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def add_message(session_id: str, role: str, content: str) -> None:
    """Append a message to an existing session."""
    now = time.time()
    with _conn() as con:
        con.execute(
            "INSERT INTO messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        con.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )


def get_messages(session_id: str) -> list[dict]:
    """Return all messages for a session, oldest first."""
    with _conn() as con:
        rows = con.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY ts ASC",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def list_sessions(limit: int = 80) -> list[dict]:
    """Return recent sessions, newest first."""
    with _conn() as con:
        rows = con.execute(
            "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]
