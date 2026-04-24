from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class ContextWindowManager:
    """Priority-based character-budget trimmer."""

    max_chars: int = 4000

    def trim(self, text: str, *, reserved_chars: int = 0) -> str:
        budget = max(self.max_chars - reserved_chars, 0)
        if len(text) <= budget:
            return text
        return text[-budget:]

    def compose(self, sections: Iterable[tuple[str, str, int]]) -> str:
        kept_sections: list[tuple[str, str, int]] = []
        remaining = self.max_chars

        for title, body, priority in sorted(sections, key=lambda item: item[2], reverse=True):
            block = f"## {title}\n{body}".strip()
            block_length = len(block) + 2
            if block_length <= remaining:
                kept_sections.append((title, body, priority))
                remaining -= block_length
                continue

            if priority >= 3 and remaining > len(title) + 8:
                trimmed_body = self.trim(body, reserved_chars=max(self.max_chars - remaining, 0))
                kept_sections.append((title, trimmed_body, priority))
                remaining = 0
                break

        return "\n\n".join(f"## {title}\n{body}" for title, body, _ in kept_sections)
