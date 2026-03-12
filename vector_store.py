import os
import time
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from utils import load_env, normalize_env_value

DEFAULT_COLLECTION = "chat_memory"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class VectorMemory:
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str | None = None,
        persist: bool = False,
        persist_dir: str | None = None,
    ) -> None:
        load_env()
        model_name = embedding_model or normalize_env_value(os.getenv("EMBEDDING_MODEL", ""))
        if not model_name:
            model_name = DEFAULT_EMBEDDING_MODEL

        if persist:
            settings = Settings(is_persistent=True, persist_directory=persist_dir or ".chroma")
        else:
            settings = Settings(is_persistent=False)

        self._embedder = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self._client = chromadb.Client(settings)
        self._collection = self._client.get_or_create_collection(
            collection_name,
            embedding_function=self._embedder,
        )

    def add_text(
        self,
        text: str,
        role: str,
        mem_type: str = "chat",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        if not text:
            return None
        doc_id = str(uuid.uuid4())
        meta = {"role": role, "type": mem_type, "ts": time.time()}
        if metadata:
            meta.update(metadata)
        self._collection.add(ids=[doc_id], documents=[text], metadatas=[meta])
        return doc_id

    def query(self, text: str, n_results: int = 4, types: list[str] | None = None) -> list[dict]:
        if not text:
            return []
        result = self._collection.query(query_texts=[text], n_results=n_results)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        records = []
        for doc, meta in zip(documents, metadatas):
            meta = meta or {}
            if types and meta.get("type") not in types:
                continue
            records.append({"text": doc, "metadata": meta})
        return records
