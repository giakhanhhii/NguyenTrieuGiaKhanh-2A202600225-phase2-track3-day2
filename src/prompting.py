from __future__ import annotations

from .context_window import ContextWindowManager
from .state import MemoryState


class PromptBuilder:
    """Build a prompt with explicit memory sections."""

    def __init__(self, trimmer: ContextWindowManager) -> None:
        self.trimmer = trimmer

    def build(self, state: MemoryState) -> str:
        sections = [
            ("Current Task", self._render_current_query(state), 4),
            ("Recent Conversation", self._render_messages(state), 4),
            ("User Profile", self._render_profile(state), 3),
            ("Episodic Memory", self._render_episodes(state), 2),
            ("Semantic Memory", self._render_semantic_hits(state), 1),
        ]
        prompt = self.trimmer.compose(sections)
        return prompt

    def _render_current_query(self, state: MemoryState) -> str:
        query = state.get("current_query", "")
        return query or "No current query."

    def _render_messages(self, state: MemoryState) -> str:
        messages = state.get("recent_conversation", []) or state.get("messages", [])
        rendered = []
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                rendered.append(f"{role}: {content}")
            else:
                rendered.append(str(message))
        return "\n".join(rendered) or "No recent messages."

    def _render_profile(self, state: MemoryState) -> str:
        profile = state.get("user_profile", {})
        if not profile:
            return "No profile facts."
        return "\n".join(f"- {key}: {value}" for key, value in profile.items())

    def _render_episodes(self, state: MemoryState) -> str:
        episodes = state.get("episodes", [])
        if not episodes:
            return "No episodic memories."
        return "\n".join(
            f"- task={episode.get('task')} outcome={episode.get('outcome')} reflection={episode.get('reflection')}"
            for episode in episodes
        )

    def _render_semantic_hits(self, state: MemoryState) -> str:
        hits = state.get("semantic_hits", [])
        if not hits:
            return "No semantic hits."
        return "\n".join(f"- {hit}" for hit in hits)
