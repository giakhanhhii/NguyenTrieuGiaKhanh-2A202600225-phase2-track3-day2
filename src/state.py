from __future__ import annotations

from typing import Any, TypedDict


class MemoryState(TypedDict, total=False):
    messages: list[Any]
    recent_conversation: list[Any]
    user_profile: dict[str, Any]
    episodes: list[dict[str, Any]]
    semantic_hits: list[str]
    memory_budget: int
    current_query: str
    selected_memories: list[str]
    prompt: str
    response: str
    profile_updates: dict[str, Any]
    episode_to_save: dict[str, Any]
