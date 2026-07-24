"""Agent-loop termination invariants (`src/retrieval/agent_grep.py`).

The loop's contract is *anytime termination*: every exit path either returns a
usable ranked candidate set or reports `finished=False`. A result that is both
`finished=True` and empty is the failure mode the protocol forbids -- it is
scored as a retrieval miss but accounted as a success, so it is invisible in the
per-cell failure rate.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backends.base import ChatResponse, ToolCall  # noqa: E402
from retrieval.agent_grep import RuleCorpus, run_agent_loop  # noqa: E402


class ScriptedBackend:
    """Replays a fixed list of ChatResponses, one per `complete` call."""

    name = "scripted"

    def __init__(self, responses: list[ChatResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, _messages, **kwargs) -> ChatResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("backend called more times than scripted")
        return self.responses.pop(0)


def messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


class AgentLoopTerminationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = RuleCorpus.from_rules()
        self.known = self.corpus.rules[0]["rule_id"]
        self.other = self.corpus.rules[1]["rule_id"]

    def grep_call(self, pattern: str = "edge") -> ChatResponse:
        return ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="grep_rules", arguments=f'{{"pattern": "{pattern}"}}')],
            finish_reason="tool_calls",
        )

    def test_truncated_submit_arguments_do_not_return_empty_as_finished(self) -> None:
        """A submit_rules call cut off mid-JSON must not yield finished=True + empty.

        `finish_reason=length` truncates the argument string, so `json.loads`
        raises and `rule_ids` is lost. The loop must recover (forced submit /
        grep-hit fallback), not report a successful empty retrieval.
        """
        truncated = ChatResponse(
            content=None,
            tool_calls=[
                ToolCall(id="c2", name="submit_rules", arguments='{"rule_ids": ["' + self.known + '", "R-CIV')
            ],
            finish_reason="length",
        )
        recovered = ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="c3", name="submit_rules", arguments='{"rule_ids": ["' + self.known + '"]}')],
            finish_reason="tool_calls",
        )
        backend = ScriptedBackend([self.grep_call(), truncated, recovered])
        result = run_agent_loop(messages(), self.corpus, backend)

        self.assertTrue(result.rule_ids, "truncated submit collapsed to an empty candidate set")
        self.assertFalse(
            result.finished and not result.rule_ids,
            "empty result must never be reported as finished=True",
        )
        self.assertIn(self.known, result.rule_ids)

    def test_submit_of_only_unknown_rule_ids_falls_back_to_grep_hits(self) -> None:
        """Hallucinated rule_ids filter to nothing; the grep evidence must survive."""
        hallucinated = ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="c2", name="submit_rules", arguments='{"rule_ids": ["R-NOPE-999"]}')],
            finish_reason="tool_calls",
        )
        recovered = ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="c3", name="submit_rules", arguments='{"rule_ids": ["' + self.known + '"]}')],
            finish_reason="tool_calls",
        )
        backend = ScriptedBackend([self.grep_call(), hallucinated, recovered])
        result = run_agent_loop(messages(), self.corpus, backend)

        self.assertTrue(
            result.retrieved,
            "submit of only-unknown ids produced an empty ranking with no recovery",
        )

    def test_total_failure_is_reported_as_unfinished_not_empty_success(self) -> None:
        """No grep evidence and a failing recovery turn must report finished=False.

        `agent_failure_rate` (experiments/retrieval/eval_visual_agent.py) counts a
        row as failed only via `error or not finished`, so an empty result marked
        finished=True is silently scored as a successful retrieval.
        """

        class ExplodingBackend:
            name = "exploding"

            def __init__(self) -> None:
                self.n = 0

            def complete(self, _messages, **kwargs) -> ChatResponse:
                self.n += 1
                if self.n == 1:
                    return ChatResponse(content="I cannot help", tool_calls=[], finish_reason="length")
                raise RuntimeError("endpoint down")

        result = run_agent_loop(messages(), self.corpus, ExplodingBackend())

        self.assertEqual([], result.rule_ids)
        self.assertFalse(
            result.finished,
            "empty candidate set reported as finished=True -> counted as success in agent_failure_rate",
        )

    def test_valid_submit_still_returns_immediately(self) -> None:
        """The happy path must not regress into an extra forced-submit turn."""
        submit = ChatResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="c2",
                    name="submit_rules",
                    arguments='{"rule_ids": ["' + self.known + '", "' + self.other + '"]}',
                )
            ],
            finish_reason="tool_calls",
        )
        backend = ScriptedBackend([submit])
        result = run_agent_loop(messages(), self.corpus, backend)

        self.assertTrue(result.finished)
        self.assertEqual([self.known, self.other], result.rule_ids)
        self.assertEqual(1, len(backend.calls), "happy path should cost exactly one call")


if __name__ == "__main__":
    unittest.main()
