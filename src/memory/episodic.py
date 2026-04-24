from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .base import MemoryBackend
from ..storage import read_json, write_json


class EpisodicMemory(MemoryBackend):
    """Episodic memory with simple similarity ranking."""

    name = "episodic"

    def __init__(self, storage_path: str | Path = "data/episodic_memory.json") -> None:
        self.storage_path = Path(storage_path)
        self._episodes: dict[str, list[dict[str, Any]]] = read_json(self.storage_path, {})

    def retrieve(self, query: str, *, user_id: str | None = None) -> list[dict[str, Any]]:
        if not user_id:
            return []
        episodes = list(self._episodes.get(user_id, []))
        if not query:
            return episodes[-3:]

        query_terms = set(re.findall(r"[a-zA-Z]+", query.lower()))
        ranked = sorted(
            episodes,
            key=lambda episode: self._score_episode(episode, query_terms),
            reverse=True,
        )
        return [episode for episode in ranked if self._score_episode(episode, query_terms) > 0][:3]

    def save(self, payload: Any, *, user_id: str | None = None) -> None:
        if not user_id or not isinstance(payload, dict):
            return
        bucket = self._episodes.setdefault(user_id, [])
        if payload not in bucket:
            bucket.append(payload)
        write_json(self.storage_path, self._episodes)

    def _score_episode(self, episode: dict[str, Any], query_terms: set[str]) -> int:
        text = " ".join(str(value) for value in episode.values()).lower()
        tokens = set(re.findall(r"[a-zA-Z]+", text))
        return sum(1 for term in query_terms if term in tokens)
