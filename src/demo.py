from __future__ import annotations

from .agent import build_agent_graph, create_default_components


def run_session(message: str, *, user_id: str) -> str:
    router, builder = create_default_components()
    graph = build_agent_graph(router=router, builder=builder, user_id=user_id)

    state = {
        "messages": [{"role": "user", "content": message}],
        "memory_budget": 1200,
    }
    result = graph.invoke(state)
    return result.get("response", "No response generated.")


def main() -> None:
    user_id = "demo-user"
    sessions = [
        ("Session 1", "I like Python and I do not like Java."),
        ("Session 2", "Can you recommend a programming language for my next agent project?"),
        ("Session 3a", "I am confused about async and await."),
        ("Session 3b", "Can you explain async/await again?"),
    ]

    for label, message in sessions:
        response = run_session(message, user_id=user_id)
        print(f"{label} user: {message}")
        print(f"{label} agent: {response}\n")


if __name__ == "__main__":
    main()
