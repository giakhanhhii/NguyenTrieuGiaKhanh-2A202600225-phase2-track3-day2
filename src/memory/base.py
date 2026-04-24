from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryBackend(ABC):
    """Common interface for all memory backends."""

    name: str

    @abstractmethod
    def retrieve(self, query: str, *, user_id: str | None = None) -> Any:
        """Load data relevant to the current query."""

    @abstractmethod
    def save(self, payload: Any, *, user_id: str | None = None) -> None:
        """Persist new memory payloads."""
