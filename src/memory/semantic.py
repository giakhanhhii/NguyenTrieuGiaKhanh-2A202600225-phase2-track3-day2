from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .base import MemoryBackend
from ..storage import read_json, write_json


class SimpleEmbeddingFunction:
    """Deterministic local embedding so Chroma works without model downloads."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 16
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(len(vector)):
                vector[index] += digest[index] / 255.0
        return vector


class SemanticMemory(MemoryBackend):
    """Semantic memory with Chroma-first storage and keyword fallback."""

    name = "semantic"

    def __init__(self, storage_path: str | Path = "data/semantic_memory.json") -> None:
        self.storage_path = Path(storage_path)
        self._snippets: list[str] = read_json(self.storage_path, [])
        self._collection = self._create_collection()

    def _create_collection(self) -> Any:
        try:
            import chromadb

            db_path = self.storage_path.parent / "chroma_db"
            client = chromadb.PersistentClient(path=str(db_path))
            return client.get_or_create_collection(
                name="semantic_memory",
                embedding_function=SimpleEmbeddingFunction(),
            )
        except Exception:
            return None

    def retrieve(self, query: str, *, user_id: str | None = None) -> list[str]:
        if not query:
            return []
        hits: list[str] = []
        if self._collection is not None:
            try:
                results = self._collection.query(query_texts=[query], n_results=3)
                docs = results.get("documents", [[]])
                hits.extend(list(docs[0]) if docs else [])
            except Exception:
                pass
        query_terms = set(self._tokenize(query))
        hits.extend(
            [
            snippet
            for snippet in self._snippets
            if query_terms.intersection(self._tokenize(snippet))
            ]
        )
        unique_hits: list[str] = []
        for hit in hits:
            if hit not in unique_hits:
                unique_hits.append(hit)
        ranked_hits = sorted(unique_hits, key=lambda hit: self._score_hit(query, hit), reverse=True)
        return ranked_hits[:3]

    def save(self, payload: Any, *, user_id: str | None = None) -> None:
        if isinstance(payload, str) and payload.strip():
            snippet = payload.strip()
            if snippet not in self._snippets:
                self._snippets.append(snippet)
                write_json(self.storage_path, self._snippets)
                if self._collection is not None:
                    try:
                        doc_id = f"semantic-{len(self._snippets)}"
                        self._collection.add(documents=[snippet], ids=[doc_id])
                    except Exception:
                        pass

    def seed(self, snippets: list[str]) -> None:
        for snippet in snippets:
            self.save(snippet)

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z]+", text.lower()))

    def _score_hit(self, query: str, hit: str) -> int:
        query_terms = self._tokenize(query)
        hit_terms = self._tokenize(hit)
        overlap = len(query_terms.intersection(hit_terms))
        starts_bonus = 2 if hit.lower().startswith("langgraph") and "langgraph" in query.lower() else 0
        password_bonus = 2 if "password" in hit.lower() and "password" in query.lower() else 0
        return overlap + starts_bonus + password_bonus
