from __future__ import annotations

import unittest
from pathlib import Path

from src.agent import build_agent_graph, create_default_components, run_turn
from src.benchmark import main as run_benchmark


class Lab17Tests(unittest.TestCase):
    def test_allergy_conflict_prefers_latest_fact(self) -> None:
        router, builder = create_default_components()
        graph = build_agent_graph(router=router, builder=builder, user_id="test-allergy")

        run_turn("I am allergic to cow milk.", user_id="test-allergy", graph=graph)
        run_turn("Actually I am allergic to soy, not cow milk.", user_id="test-allergy", graph=graph)
        result = run_turn("What is my allergy?", user_id="test-allergy", graph=graph)

        self.assertIn("soy", result.get("response", "").lower())

    def test_async_episode_is_recalled(self) -> None:
        router, builder = create_default_components()
        graph = build_agent_graph(router=router, builder=builder, user_id="test-episode")

        run_turn("I am confused about async and await.", user_id="test-episode", graph=graph)
        result = run_turn("Can you explain async/await again?", user_id="test-episode", graph=graph)

        self.assertIn("last time async/await felt confusing", result.get("response", "").lower())

    def test_benchmark_generates_outputs(self) -> None:
        run_benchmark()
        self.assertTrue(Path("BENCHMARK.md").exists())
        self.assertTrue(Path("benchmark/results.json").exists())
        self.assertTrue(Path("benchmark/report.md").exists())


if __name__ == "__main__":
    unittest.main()
