from __future__ import annotations

from collections import deque
from typing import Any

from .base import MemoryBackend


class ShortTermMemory(MemoryBackend):
    """Conversation buffer skeleton."""

    name = "short_term"

    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max_turns
        self._buffer: deque[Any] = deque(maxlen=max_turns)

    def retrieve(self, query: str, *, user_id: str | None = None) -> list[Any]:
        return list(self._buffer)

    def save(self, payload: Any, *, user_id: str | None = None) -> None:
        self._buffer.append(payload)
