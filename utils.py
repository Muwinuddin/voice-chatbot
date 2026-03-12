import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path, override=True)


def normalize_env_value(value: str) -> str:
    return value.strip().strip('"').strip("'")


def extract_profile_facts(text: str) -> list[str]:
    cleaned = clean_text(text)
    lower = cleaned.lower()
    triggers = [
        ("my name is", "Name"),
        ("i live in", "Location"),
        ("i am from", "Origin"),
        ("my favorite", "Favorite"),
        ("my email is", "Email"),
    ]
    facts = []
    for trigger, label in triggers:
        if trigger in lower:
            start = lower.find(trigger) + len(trigger)
            snippet = cleaned[start:].strip()
            snippet = snippet.split(".")[0].split(",")[0]
            snippet = " ".join(snippet.split()[:12])
            if snippet:
                facts.append(f"{label}: {snippet}")
    return facts


def format_memory_snippets(records: list[dict]) -> str:
    if not records:
        return ""
    lines = []
    for record in records:
        meta = record.get("metadata") or {}
        role = meta.get("role", "memory")
        mem_type = meta.get("type", "memory")
        text = record.get("text", "")
        if text:
            lines.append(f"{role} ({mem_type}): {text}")
    return "\n".join(lines)


def format_search_results(results: list[dict]) -> str:
    if not results:
        return "No search results found."
    lines = []
    for idx, item in enumerate(results, start=1):
        title = item.get("title", "Result")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        line = f"{idx}. {title} - {snippet} ({link})"
        lines.append(line.strip())
    return "\n".join(lines)


def save_audio_file(audio_bytes: bytes, suffix: str = ".mp3") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as handle:
        handle.write(audio_bytes)
    return path
