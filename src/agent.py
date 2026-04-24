from __future__ import annotations

import re
from typing import Any

from .context_window import ContextWindowManager
from .memory import EpisodicMemory, ProfileMemory, SemanticMemory, ShortTermMemory
from .prompting import PromptBuilder
from .router import MemoryRouter
from .state import MemoryState

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - allows scaffold use before install
    END = "__end__"
    START = "__start__"
    StateGraph = None


class LocalCompiledGraph:
    """Fallback graph runner when LangGraph is unavailable."""

    def __init__(self, *, router: MemoryRouter, builder: PromptBuilder, user_id: str) -> None:
        self.router = router
        self.builder = builder
        self.user_id = user_id

    def invoke(self, state: MemoryState) -> MemoryState:
        merged: MemoryState = dict(state)
        merged.update(retrieve_memory(merged, router=self.router, user_id=self.user_id))
        merged.update(build_prompt(merged, builder=self.builder))
        merged.update(generate_response(merged))
        merged.update(save_memory(merged, router=self.router, user_id=self.user_id))
        return merged


def create_default_components() -> tuple[MemoryRouter, PromptBuilder]:
    short_term = ShortTermMemory()
    profile = ProfileMemory()
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    semantic.seed(
        [
            "Python is often a strong choice for agent systems because the ecosystem around LangGraph and orchestration is mature.",
            "Java is powerful for enterprise systems, but it is usually less lightweight for rapid agent prototyping than Python.",
            "Async and await let Python pause work on I/O and resume later without blocking the whole program.",
            "LangGraph is a framework for building stateful graph-based agent workflows in Python.",
            "When debugging Docker Compose networking, use the service name instead of localhost between containers.",
            "The FAQ says password resets should be done from Settings > Security > Reset Password.",
        ]
    )
    router = MemoryRouter(short_term, profile, episodic, semantic)
    builder = PromptBuilder(ContextWindowManager())
    return router, builder


def retrieve_memory(state: MemoryState, *, router: MemoryRouter, user_id: str) -> MemoryState:
    return router.retrieve_memory(state, user_id=user_id)


def save_memory(state: MemoryState, *, router: MemoryRouter, user_id: str) -> MemoryState:
    messages = state.get("messages", [])
    if messages:
        router.short_term.save(messages[-1], user_id=user_id)
    response = state.get("response")
    if response:
        router.short_term.save({"role": "assistant", "content": response}, user_id=user_id)

    profile_updates = state.get("profile_updates", {})
    if profile_updates:
        router.profile.save(profile_updates, user_id=user_id)

    episode_to_save = state.get("episode_to_save")
    if episode_to_save:
        router.episodic.save(episode_to_save, user_id=user_id)
    return state


def build_prompt(state: MemoryState, *, builder: PromptBuilder) -> MemoryState:
    return {"prompt": builder.build(state)}


def generate_response(state: MemoryState) -> MemoryState:
    query = state.get("current_query", "")
    lowered = query.lower()
    profile = state.get("user_profile", {})
    episodes = state.get("episodes", [])
    semantic_hits = state.get("semantic_hits", [])
    profile_updates = extract_profile_updates(query)
    episode_to_save = detect_episode(query)

    if "recommend" in lowered and profile.get("likes_language") == "Python":
        response = (
            "I still recommend Python for you. You previously said you like Python"
            " and dislike Java, so Python is the better fit for this task."
        )
    elif ("what is my name" in lowered or "remember my name" in lowered) and profile.get("name"):
        response = f"Your name is {profile['name']}."
    elif ("what style" in lowered or "response style" in lowered or "how should you answer" in lowered) and profile.get("response_style"):
        response = f"You asked for a {profile['response_style']} response style."
    elif ("what is my allergy" in lowered or "what allergy" in lowered) and profile.get("allergy"):
        response = f"Your allergy profile is currently set to {profile['allergy']}."
    elif "confused" in lowered and ("async" in lowered or "await" in lowered):
        response = (
            "Thanks for telling me that async/await feels confusing. "
            "I will explain it more carefully: `async` marks a coroutine, "
            "and `await` pauses that coroutine until an I/O task finishes."
        )
    elif ("what worked last time" in lowered or "how did we solve" in lowered) and episodes:
        latest = episodes[0]
        response = (
            f"Last time the useful lesson was: {latest.get('reflection', 'No reflection stored.')}"
        )
    elif "async" in lowered and episodes:
        response = (
            "Last time async/await felt confusing, so here is the simpler version: "
            "async defines a coroutine, await pauses inside that coroutine until an I/O task finishes, "
            "and the event loop can work on other tasks in the meantime."
        )
    elif "like python" in lowered or "don't like java" in lowered or "do not like java" in lowered:
        response = "Noted. I will remember that you like Python and do not like Java."
    elif "allergic" in lowered or "dị ứng" in lowered:
        allergy = profile_updates.get("allergy", "your latest allergy info")
        response = f"Understood. I will keep your allergy profile updated as {allergy}."
    elif semantic_hits:
        response = f"Here is the most relevant memory I found: {semantic_hits[0]}"
    else:
        response = (
            "I have gathered the most relevant memories and built the prompt context. "
            "The next step is connecting this flow to a real LLM call."
        )

    return {
        "response": response,
        "profile_updates": profile_updates,
        "episode_to_save": episode_to_save,
    }


def extract_profile_updates(query: str) -> dict[str, str]:
    lowered = query.lower()
    updates: dict[str, str] = {}

    if "like python" in lowered or "prefer python" in lowered:
        updates["likes_language"] = "Python"
    if "dislike java" in lowered or "don't like java" in lowered or "do not like java" in lowered:
        updates["dislikes_language"] = "Java"
    name_match = re.search(r"\b(?:my name is|call me)\s+([a-zA-Z][a-zA-Z ]+)", query, flags=re.IGNORECASE)
    if name_match:
        updates["name"] = name_match.group(1).strip().split(".")[0].strip().title()
    if "prefer concise" in lowered or "keep answers concise" in lowered:
        updates["response_style"] = "concise"
    if "prefer detailed" in lowered or "want detailed" in lowered:
        updates["response_style"] = "detailed"

    allergy_match = re.search(r"allergic to ([a-zA-Z ]+)", lowered)
    if allergy_match:
        updates["allergy"] = allergy_match.group(1).strip()
    if "soy" in lowered or "đậu nành" in lowered:
        updates["allergy"] = "soy"
    if "cow milk" in lowered or "sữa bò" in lowered:
        updates.setdefault("allergy", "cow milk")

    return updates


def detect_episode(query: str) -> dict[str, str] | None:
    lowered = query.lower()
    if "confused" in lowered and ("async" in lowered or "await" in lowered):
        return {
            "task": "understand async/await",
            "outcome": "user was confused",
            "reflection": "Explain async/await more simply next time.",
        }
    if "docker service name" in lowered or "use the service name" in lowered:
        return {
            "task": "debug docker networking",
            "outcome": "issue solved",
            "reflection": "Use the Docker service name instead of localhost between containers.",
        }
    return None


def build_agent_graph(*, router: MemoryRouter, builder: PromptBuilder, user_id: str) -> Any:
    if StateGraph is None:
        return LocalCompiledGraph(router=router, builder=builder, user_id=user_id)

    graph = StateGraph(MemoryState)
    graph.add_node("retrieve_memory", lambda state: retrieve_memory(state, router=router, user_id=user_id))
    graph.add_node("build_prompt", lambda state: build_prompt(state, builder=builder))
    graph.add_node("generate_response", generate_response)
    graph.add_node("save_memory", lambda state: save_memory(state, router=router, user_id=user_id))

    graph.add_edge(START, "retrieve_memory")
    graph.add_edge("retrieve_memory", "build_prompt")
    graph.add_edge("build_prompt", "generate_response")
    graph.add_edge("generate_response", "save_memory")
    graph.add_edge("save_memory", END)
    return graph.compile()


def run_turn(message: str, *, user_id: str, graph: Any) -> MemoryState:
    state: MemoryState = {
        "messages": [{"role": "user", "content": message}],
        "memory_budget": 1200,
    }
    return graph.invoke(state)
