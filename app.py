import os
import time

import requests
import streamlit as st

from groq_client import get_groq_response
from history_store import (
    add_message,
    create_session,
    delete_session,
    get_messages,
    list_sessions,
    rename_session,
)
from search_client import search_web
from tts_client import speak_text
from utils import (
    clean_text,
    extract_profile_facts,
    format_memory_snippets,
    format_search_results,
    load_env,
    normalize_env_value,
)
from rag_store import RAGStore
from vector_store import VectorMemory

# ── env ──────────────────────────────────────────────────────────────────────
load_env()
BACKEND_URL = normalize_env_value(os.getenv("BACKEND_URL", ""))
USE_BACKEND = bool(BACKEND_URL)
memory_store: VectorMemory | None = None
rag_store: RAGStore | None = None
if not USE_BACKEND:
    memory_store = VectorMemory()
    rag_store = RAGStore()

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Voice Chatbot", layout="wide", page_icon="🎙️")

# ── session state defaults ────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "history" not in st.session_state:
    st.session_state.history = []
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "delete_confirm" not in st.session_state:
    st.session_state.delete_confirm = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _start_new_session() -> None:
    sid = create_session("New chat")
    st.session_state.session_id = sid
    st.session_state.history = []
    st.session_state.last_audio = None
    st.session_state.delete_confirm = None


def _load_session(sid: str) -> None:
    st.session_state.session_id = sid
    st.session_state.history = get_messages(sid)
    st.session_state.last_audio = None
    st.session_state.delete_confirm = None


def _fmt_time(ts: float) -> str:
    diff = time.time() - ts
    if diff < 60:
        return "just now"
    if diff < 3600:
        return f"{int(diff // 60)}m ago"
    if diff < 86400:
        return f"{int(diff // 3600)}h ago"
    return time.strftime("%b %d", time.localtime(ts))


def should_search(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    keywords = [
        "search", "web", "internet", "latest", "news", "current", "today",
        "recent", "lookup", "find", "price", "stock", "weather", "score",
    ]
    if any(word in t for word in keywords):
        return True
    return "?" in t and any(w in t for w in ["who", "what", "when", "where", "why", "how"])


def build_prompt(history, search_context: str, memory_context: str, rag_context: str = "") -> str:
    lines = ["You are a helpful voice assistant. Keep answers clear and concise."]
    if memory_context:
        lines += ["Relevant memory:", memory_context]
    if rag_context:
        lines.append(rag_context)
    if search_context:
        lines += ["Use these web search results when relevant:", search_context]
    lines.append("Conversation:")
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


def backend_chat(message: str, history: list[dict], use_search: bool) -> dict:
    url = f"{BACKEND_URL.rstrip('/')}/chat"
    r = requests.post(url, json={"message": message, "history": history, "use_search": use_search}, timeout=30)
    r.raise_for_status()
    return r.json()


def backend_tts(text: str) -> bytes:
    r = requests.post(f"{BACKEND_URL.rstrip('/')}/tts", json={"text": text}, timeout=30)
    r.raise_for_status()
    return r.content


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎙️ Voice Chatbot")
    st.divider()

    if st.button("➕  New chat", use_container_width=True, type="primary"):
        _start_new_session()
        st.rerun()

    st.subheader("Chat History")
    sessions = list_sessions(limit=80)

    if not sessions:
        st.caption("No past sessions yet.")
    else:
        today_ts = time.time()
        groups: dict[str, list[dict]] = {}
        for s in sessions:
            diff = today_ts - s["updated_at"]
            if diff < 86400:
                g_label = "Today"
            elif diff < 172800:
                g_label = "Yesterday"
            elif diff < 604800:
                g_label = "This week"
            else:
                g_label = time.strftime("%B %Y", time.localtime(s["updated_at"]))
            groups.setdefault(g_label, []).append(s)

        active = st.session_state.session_id
        for g_label, g_sessions in groups.items():
            st.caption(g_label)
            for s in g_sessions:
                is_active = s["id"] == active
                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    label_text = ("► " if is_active else "   ") + s["title"]
                    if st.button(
                        label_text,
                        key=f"sess_{s['id']}",
                        use_container_width=True,
                        help=f"Last active {_fmt_time(s['updated_at'])}",
                    ):
                        _load_session(s["id"])
                        st.rerun()
                with col_del:
                    if st.button("🗑", key=f"del_{s['id']}", help="Delete session"):
                        st.session_state.delete_confirm = s["id"]
                        st.rerun()

        if st.session_state.delete_confirm:
            sid_del = st.session_state.delete_confirm
            title_del = next((s["title"] for s in sessions if s["id"] == sid_del), "this chat")
            st.warning(f'Delete "{title_del}"?')
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, delete", type="primary", key="confirm_del"):
                    delete_session(sid_del)
                    if st.session_state.session_id == sid_del:
                        st.session_state.session_id = None
                        st.session_state.history = []
                    st.session_state.delete_confirm = None
                    st.rerun()
            with c2:
                if st.button("Cancel", key="cancel_del"):
                    st.session_state.delete_confirm = None
                    st.rerun()

    # ── RAG knowledge base panel ─────────────────────────────────────
    st.divider()
    st.subheader("📚 Knowledge Base (RAG)")

    rag_tab_ingest, rag_tab_docs = st.tabs(["Ingest", "Documents"])

    with rag_tab_ingest:
        with st.form("rag_ingest_form", clear_on_submit=True):
            rag_source_name = st.text_input("Document name", placeholder="e.g. company-faq")
            rag_text_input = st.text_area("Paste text", height=140, placeholder="Paste document content here…")
            rag_pdf_file = st.file_uploader("or upload a PDF", type=["pdf"])
            rag_submitted = st.form_submit_button("Ingest ➕", use_container_width=True)

        if rag_submitted:
            if not rag_source_name:
                st.warning("Please provide a document name.")
            elif not rag_text_input and rag_pdf_file is None:
                st.warning("Paste text or upload a PDF.")
            elif USE_BACKEND:
                try:
                    if rag_pdf_file is not None:
                        resp = requests.post(
                            f"{BACKEND_URL.rstrip('/')}/rag/ingest/pdf",
                            files={"file": (rag_pdf_file.name, rag_pdf_file.getvalue(), "application/pdf")},
                            timeout=30,
                        )
                    else:
                        resp = requests.post(
                            f"{BACKEND_URL.rstrip('/')}/rag/ingest/text",
                            json={"text": rag_text_input, "source_name": rag_source_name},
                            timeout=30,
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    st.success(f"Ingested {data['chunk_count']} chunks from '{data['source_name']}'.")
                except Exception as exc:
                    st.error(f"Ingest failed: {exc}")
            elif rag_store is not None:
                try:
                    if rag_pdf_file is not None:
                        source_id = rag_store.ingest_pdf_bytes(rag_pdf_file.getvalue(), rag_source_name)
                    else:
                        source_id = rag_store.ingest_text(rag_text_input, rag_source_name)
                    sources_map = {s["id"]: s for s in rag_store.list_sources()}
                    count = sources_map.get(source_id, {}).get("chunk_count", 0)
                    st.success(f"Ingested {count} chunks from '{rag_source_name}'.")
                except Exception as exc:
                    st.error(f"Ingest failed: {exc}")

    with rag_tab_docs:
        if USE_BACKEND:
            try:
                resp = requests.get(f"{BACKEND_URL.rstrip('/')}/rag/sources", timeout=10)
                resp.raise_for_status()
                rag_sources = resp.json()
            except Exception:
                rag_sources = []
        elif rag_store is not None:
            rag_sources = rag_store.list_sources()
        else:
            rag_sources = []

        if not rag_sources:
            st.caption("No documents ingested yet.")
        else:
            for src in rag_sources:
                c_name, c_del = st.columns([4, 1])
                with c_name:
                    st.markdown(f"**{src['name']}**  `{src['chunk_count']} chunks`")
                with c_del:
                    if st.button("🗑", key=f"ragdel_{src['id']}", help="Delete"):
                        if USE_BACKEND:
                            try:
                                requests.delete(
                                    f"{BACKEND_URL.rstrip('/')}/rag/sources/{src['id']}",
                                    timeout=10,
                                )
                            except Exception as exc:
                                st.error(f"Delete failed: {exc}")
                        elif rag_store is not None:
                            rag_store.delete_source(src["id"])
                        st.rerun()

    st.divider()
    if USE_BACKEND:
        st.caption(f"Backend: {BACKEND_URL}")
    else:
        st.caption("Mode: local (Groq + Azure TTS + Chroma + RAG)")

# ── main area ─────────────────────────────────────────────────────────────────
PENGUIN_SVG = """
<div style="display:flex;align-items:center;gap:18px;margin-bottom:8px;">
  <svg xmlns="http://www.w3.org/2000/svg" width="90" height="110" viewBox="0 0 90 110">
    <!-- body -->
    <ellipse cx="45" cy="72" rx="28" ry="34" fill="#1a1a2e"/>
    <!-- belly -->
    <ellipse cx="45" cy="78" rx="16" ry="22" fill="#f5f0e8"/>
    <!-- head -->
    <ellipse cx="45" cy="34" rx="22" ry="22" fill="#1a1a2e"/>
    <!-- face white patch -->
    <ellipse cx="45" cy="36" rx="13" ry="14" fill="#f5f0e8"/>
    <!-- left eye -->
    <circle cx="39" cy="30" r="4.5" fill="white"/>
    <circle cx="40" cy="30" r="2.5" fill="#111"/>
    <circle cx="40.8" cy="29.2" r="0.9" fill="white"/>
    <!-- right eye -->
    <circle cx="51" cy="30" r="4.5" fill="white"/>
    <circle cx="52" cy="30" r="2.5" fill="#111"/>
    <circle cx="52.8" cy="29.2" r="0.9" fill="white"/>
    <!-- beak -->
    <ellipse cx="45" cy="40" rx="5" ry="3.5" fill="#f4a020"/>
    <!-- left wing -->
    <ellipse cx="18" cy="72" rx="9" ry="22" fill="#1a1a2e" transform="rotate(-10 18 72)"/>
    <!-- right wing -->
    <ellipse cx="72" cy="72" rx="9" ry="22" fill="#1a1a2e" transform="rotate(10 72 72)"/>
    <!-- left foot -->
    <ellipse cx="35" cy="104" rx="10" ry="5" fill="#f4a020" transform="rotate(-15 35 104)"/>
    <!-- right foot -->
    <ellipse cx="55" cy="104" rx="10" ry="5" fill="#f4a020" transform="rotate(15 55 104)"/>
    <!-- headphone band -->
    <path d="M23 34 Q45 10 67 34" stroke="#4a90d9" stroke-width="4" fill="none" stroke-linecap="round"/>
    <!-- headphone left cup -->
    <rect x="17" y="32" width="12" height="10" rx="5" fill="#4a90d9"/>
    <!-- headphone right cup -->
    <rect x="61" y="32" width="12" height="10" rx="5" fill="#4a90d9"/>
    <!-- mic squiggle on belly -->
    <text x="38" y="76" font-size="13" fill="#4a90d9" font-family="Arial">🎙</text>
  </svg>
  <div>
    <h1 style="margin:0;font-size:2rem;font-weight:800;color:#1a1a2e;">Voice Chatbot</h1>
    <p style="margin:4px 0 0;color:#555;font-size:0.95rem;">
      Type a message or record your voice · Web search · RAG knowledge base
    </p>
  </div>
</div>
"""
st.markdown(PENGUIN_SVG, unsafe_allow_html=True)

if st.session_state.session_id is None:
    _start_new_session()




# Render existing conversation
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.last_audio:
    st.audio(st.session_state.last_audio, format="audio/mpeg")

# ── input form ────────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_input("Your message", placeholder="Ask me anything…")
    mic_audio = st.audio_input("🎤 Record voice message")
    submitted = st.form_submit_button("Send ➤", use_container_width=True)

if mic_audio is not None and not submitted:
    st.audio(mic_audio)
    st.info("Audio recorded. Type a message to chat, or enable transcription.")

search_results: list[dict] = []
if submitted and user_text:
    cleaned = clean_text(user_text)
    sid = st.session_state.session_id

    # Auto-title the session from the first user message
    if not st.session_state.history:
        rename_session(sid, cleaned[:60])

    st.session_state.history.append({"role": "user", "content": cleaned})
    add_message(sid, "user", cleaned)

    if USE_BACKEND:
        try:
            result = backend_chat(cleaned, st.session_state.history, use_search=should_search(cleaned))
            response_text = result.get("response", "").strip()
            search_results = result.get("search_results") or []
        except Exception as exc:
            st.error(f"Backend error: {exc}")
            response_text = "Sorry, I ran into an error."
    else:
        search_context = ""
        if should_search(cleaned):
            try:
                search_results = search_web(cleaned)
                search_context = format_search_results(search_results)
            except Exception as exc:
                st.warning(f"Search failed: {exc}")

        memory_context = ""
        if memory_store:
            records = memory_store.query(cleaned, n_results=4, types=["chat", "profile"])
            memory_context = format_memory_snippets(records)

        rag_context = ""
        if rag_store:
            rag_records = rag_store.query(cleaned, n_results=4)
            rag_context = rag_store.format_context(rag_records)

        try:
            prompt = build_prompt(st.session_state.history, search_context, memory_context, rag_context)
            response_text = get_groq_response(prompt)
        except Exception as exc:
            st.error(f"Groq API error: {exc}")
            response_text = "Sorry, I ran into an error."

    st.session_state.history.append({"role": "assistant", "content": response_text})
    add_message(sid, "assistant", response_text)

    if memory_store:
        memory_store.add_text(cleaned, role="user", mem_type="chat")
        for fact in extract_profile_facts(cleaned):
            memory_store.add_text(fact, role="user", mem_type="profile")
        memory_store.add_text(response_text, role="assistant", mem_type="chat")

    try:
        st.session_state.last_audio = backend_tts(response_text) if USE_BACKEND else speak_text(response_text)
    except Exception as exc:
        st.session_state.last_audio = None
        st.warning(f"Audio generation failed: {exc}")

    st.rerun()

# ── search results expander ───────────────────────────────────────────────────
if search_results:
    with st.expander("🔍 Web search results", expanded=False):
        for item in search_results:
            title = item.get("title") or "Result"
            link = item.get("link") or ""
            snippet = item.get("snippet") or ""
            st.markdown(f"**[{title}]({link})**" if link else f"**{title}**")
            if snippet:
                st.caption(snippet)
            st.divider()

