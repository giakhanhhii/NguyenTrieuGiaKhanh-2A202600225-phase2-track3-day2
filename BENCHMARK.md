# BENCHMARK

Comparison of `no-memory` vs `with-memory` on 10 multi-turn conversations.

| # | Scenario | No-memory result | With-memory result | Pass? |
|---|----------|------------------|--------------------|-------|
| 1 | Recall user name after several turns | I do not know because this run does not retain memory across turns. | Your name is Linh. | Pass |
| 2 | Language preference recall | Python is a common default choice, but I am not using stored preferences here. | I still recommend Python for you. You previously said you like Python and dislike Java, so Python is the better fit for this task. | Pass |
| 3 | Allergy conflict update | I do not know because this run does not retain memory across turns. | Your allergy profile is currently set to soy. | Pass |
| 4 | Response style recall | I do not know because this run does not retain memory across turns. | You asked for a concise response style. | Pass |
| 5 | Async confusion episodic recall | Async and await are Python features for asynchronous code. | Last time async/await felt confusing, so here is the simpler version: async defines a coroutine, await pauses inside that coroutine until an I/O task finishes, and the event loop can work on other tasks in the meantime. | Pass |
| 6 | Docker debug lesson recall | I do not know because this run does not retain past episodes. | Last time the useful lesson was: Use the Docker service name instead of localhost between containers. | Pass |
| 7 | Semantic retrieval for LangGraph | LangGraph helps build agent workflows. | Here is the most relevant memory I found: LangGraph is a framework for building stateful graph-based agent workflows in Python. | Pass |
| 8 | Semantic retrieval for FAQ chunk | I do not have stored FAQ memory in no-memory mode. | Here is the most relevant memory I found: The FAQ says password resets should be done from Settings > Security > Reset Password. | Pass |
| 9 | Trim and token budget on long conversation | I do not know because this run does not retain memory across turns. | Your name is Linh. | Pass |
| 10 | Detailed style preference recall | I do not know because this run does not retain memory across turns. | You asked for a detailed response style. | Pass |

## Memory Hit-Rate Analysis

- Passed scenarios: 10/10
- Profile recall passes: 4
- Conflict update passes: 1
- Episodic recall passes: 2
- Semantic retrieval passes: 2
- Trim/token budget passes: 1

## Token Budget Breakdown

- Scenario 1: with-memory prompt chars = 1205
- Scenario 2: with-memory prompt chars = 596
- Scenario 3: with-memory prompt chars = 713
- Scenario 4: with-memory prompt chars = 572
- Scenario 5: with-memory prompt chars = 776
- Scenario 6: with-memory prompt chars = 743
- Scenario 7: with-memory prompt chars = 507
- Scenario 8: with-memory prompt chars = 399
- Scenario 9: with-memory prompt chars = 1763
- Scenario 10: with-memory prompt chars = 570

## Interpretation

- Profile memory helps on name, allergy, language, and response-style recall.
- Episodic memory helps on recalling previous confusion and previous debugging outcomes.
- Semantic memory helps on factual retrieval even when the user asks only one turn.
- The trim/token-budget case checks that prompt construction stays inside a bounded context size.
