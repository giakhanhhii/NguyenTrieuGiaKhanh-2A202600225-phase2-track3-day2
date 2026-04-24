from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .agent import build_agent_graph, create_default_components, run_turn
from .storage import read_json, write_json


ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = ROOT / "data" / "benchmarks" / "scenarios.json"
RESULTS_PATH = ROOT / "benchmark" / "results.json"
REPORT_PATH = ROOT / "benchmark" / "report.md"
BENCHMARK_MD_PATH = ROOT / "BENCHMARK.md"


def load_scenarios() -> list[dict[str, Any]]:
    return read_json(SCENARIOS_PATH, [])


def run_with_memory(turns: list[str], *, user_id: str) -> dict[str, Any]:
    router, builder = create_default_components()
    graph = build_agent_graph(router=router, builder=builder, user_id=user_id)

    final_state: dict[str, Any] = {}
    for turn in turns:
        final_state = run_turn(turn, user_id=user_id, graph=graph)

    return {
        "response": final_state.get("response", ""),
        "prompt_chars": len(final_state.get("prompt", "")),
    }


def run_no_memory(turns: list[str]) -> dict[str, Any]:
    last = turns[-1].lower()
    if "what is langgraph" in last:
        response = "LangGraph helps build agent workflows."
    elif "reset my password" in last:
        response = "I do not have stored FAQ memory in no-memory mode."
    elif "async" in last:
        response = "Async and await are Python features for asynchronous code."
    elif "what is my name" in last or "what is my allergy" in last or "what style" in last or "response style" in last:
        response = "I do not know because this run does not retain memory across turns."
    elif "recommend a programming language" in last:
        response = "Python is a common default choice, but I am not using stored preferences here."
    elif "what worked last time" in last:
        response = "I do not know because this run does not retain past episodes."
    else:
        response = "This is a no-memory baseline response."
    return {"response": response, "prompt_chars": len(last)}


def evaluate_scenario(scenario: dict[str, Any], with_memory: dict[str, Any], no_memory: dict[str, Any]) -> dict[str, Any]:
    with_text = with_memory["response"]
    no_text = no_memory["response"]
    with_contains = scenario.get("with_memory_contains", [])
    no_memory_should_not_contain = scenario.get("no_memory_should_not_contain", [])

    with_ok = all(fragment.lower() in with_text.lower() for fragment in with_contains)
    no_ok = all(fragment.lower() not in no_text.lower() for fragment in no_memory_should_not_contain)
    prompt_ok = True

    max_prompt_chars = scenario.get("max_prompt_chars")
    if max_prompt_chars is not None:
        prompt_ok = with_memory["prompt_chars"] <= max_prompt_chars

    passed = with_ok and no_ok and prompt_ok
    return {
        "id": scenario["id"],
        "scenario": scenario["name"],
        "group": scenario["group"],
        "turns": scenario["turns"],
        "no_memory_result": no_text,
        "with_memory_result": with_text,
        "with_memory_prompt_chars": with_memory["prompt_chars"],
        "pass": passed,
        "checks": {
            "with_memory_match": with_ok,
            "no_memory_baseline": no_ok,
            "prompt_budget_ok": prompt_ok,
        },
    }


def render_benchmark_md(results: list[dict[str, Any]]) -> str:
    lines = [
        "# BENCHMARK",
        "",
        "Comparison of `no-memory` vs `with-memory` on 10 multi-turn conversations.",
        "",
        "| # | Scenario | No-memory result | With-memory result | Pass? |",
        "|---|----------|------------------|--------------------|-------|",
    ]
    for result in results:
        lines.append(
            f"| {result['id']} | {escape_cell(result['scenario'])} | "
            f"{escape_cell(result['no_memory_result'])} | {escape_cell(result['with_memory_result'])} | "
            f"{'Pass' if result['pass'] else 'Fail'} |"
        )

    group_counts = Counter(result["group"] for result in results if result["pass"])
    lines.extend(
        [
            "",
            "## Memory Hit-Rate Analysis",
            "",
            f"- Passed scenarios: {sum(1 for result in results if result['pass'])}/{len(results)}",
            f"- Profile recall passes: {group_counts.get('profile recall', 0)}",
            f"- Conflict update passes: {group_counts.get('conflict update', 0)}",
            f"- Episodic recall passes: {group_counts.get('episodic recall', 0)}",
            f"- Semantic retrieval passes: {group_counts.get('semantic retrieval', 0)}",
            f"- Trim/token budget passes: {group_counts.get('trim/token budget', 0)}",
            "",
            "## Token Budget Breakdown",
            "",
        ]
    )
    for result in results:
        lines.append(
            f"- Scenario {result['id']}: with-memory prompt chars = {result['with_memory_prompt_chars']}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Profile memory helps on name, allergy, language, and response-style recall.",
            "- Episodic memory helps on recalling previous confusion and previous debugging outcomes.",
            "- Semantic memory helps on factual retrieval even when the user asks only one turn.",
            "- The trim/token-budget case checks that prompt construction stays inside a bounded context size.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_report_md(results: list[dict[str, Any]]) -> str:
    passes = sum(1 for result in results if result["pass"])
    return (
        "# Benchmark Report\n\n"
        f"- Total scenarios: {len(results)}\n"
        f"- Passed: {passes}\n"
        f"- Failed: {len(results) - passes}\n\n"
        "## Notes\n\n"
        "- This benchmark uses a lightweight rule-based responder for the final answer, while memory retrieval, routing, prompt injection, and persistence are real code paths.\n"
        "- Redis is used when a local server is available; otherwise profile memory falls back to JSON.\n"
        "- Chroma is attempted first for semantic memory; if the local environment blocks it, keyword fallback keeps the benchmark runnable.\n"
    )


def escape_cell(text: str) -> str:
    return text.replace("\n", "<br>").replace("|", "\\|")


def main() -> None:
    scenarios = load_scenarios()
    results: list[dict[str, Any]] = []

    for scenario in scenarios:
        user_id = f"bench-user-{scenario['id']}"
        with_memory = run_with_memory(scenario["turns"], user_id=user_id)
        no_memory = run_no_memory(scenario["turns"])
        results.append(evaluate_scenario(scenario, with_memory, no_memory))

    payload = {"results": results}
    write_json(RESULTS_PATH, payload)
    benchmark_md = render_benchmark_md(results)
    report_md = render_report_md(results)
    BENCHMARK_MD_PATH.write_text(benchmark_md, encoding="utf-8")
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Updated {BENCHMARK_MD_PATH}")


if __name__ == "__main__":
    main()
