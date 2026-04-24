from __future__ import annotations

from typing import Any

from .memory import EpisodicMemory, ProfileMemory, SemanticMemory, ShortTermMemory
from .state import MemoryState


class MemoryRouter:
    """Aggregate memory from multiple backends into agent state."""

    def __init__(
        self,
        short_term: ShortTermMemory,
        profile: ProfileMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
    ) -> None:
        self.short_term = short_term
        self.profile = profile
        self.episodic = episodic
        self.semantic = semantic

    def retrieve_memory(self, state: MemoryState, *, user_id: str) -> MemoryState:
        messages = state.get("messages", [])
        latest = messages[-1] if messages else ""
        query = self._message_to_text(latest)
        selected_memories = self.route(query)
        recent_conversation = self.short_term.retrieve(query, user_id=user_id)

        profile = self.profile.retrieve(query, user_id=user_id) if "profile" in selected_memories else {}
        episodes = self.episodic.retrieve(query, user_id=user_id) if "episodic" in selected_memories else []
        semantic_hits = self.semantic.retrieve(query, user_id=user_id) if "semantic" in selected_memories else []

        return {
            "messages": messages,
            "recent_conversation": recent_conversation,
            "current_query": query,
            "selected_memories": selected_memories,
            "user_profile": profile,
            "episodes": episodes,
            "semantic_hits": semantic_hits,
            "memory_budget": state.get("memory_budget", 1200),
        }

    def route(self, query: str) -> list[str]:
        lowered = query.lower()
        selected = ["short_term"]

        preference_keywords = [
            "like",
            "dislike",
            "prefer",
            "favorite",
            "recommend",
            "language",
            "allerg",
            "hate",
            "my name",
            "response style",
            "style should",
            "what style",
            "what is my allergy",
        ]
        episodic_keywords = [
            "before",
            "last time",
            "previous",
            "remember",
            "confused",
            "again",
            "earlier",
            "stuck",
            "async",
            "await",
        ]
        semantic_keywords = [
            "what is",
            "explain",
            "how",
            "faq",
            "semantic",
            "context",
            "memory",
            "langgraph",
            "async",
            "await",
            "password",
            "reset",
        ]

        if any(keyword in lowered for keyword in preference_keywords):
            selected.append("profile")
        if any(keyword in lowered for keyword in episodic_keywords):
            selected.append("episodic")
        if any(keyword in lowered for keyword in semantic_keywords):
            selected.append("semantic")

        if len(selected) == 1:
            selected.extend(["profile", "semantic"])
        return selected

    def _message_to_text(self, message: Any) -> str:
        if isinstance(message, str):
            return message
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(message)
