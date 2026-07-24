"""Ranked grep (CS_GREP_RANK=1) orders hits by match strength, not corpus order.

The default grep keeps the first `max_hits` rules in corpus order; at 42 rules a
specific pattern rarely overflows, so which rules survive truncation is an
accident of ordering. The ranked variant scores each hit by field weight x match
count so the strongest survive. These tests are hermetic: they build a small
`RuleCorpus` from synthetic rules with controlled fields, so the assertions do
not depend on the live rule library.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CS_ROOT / "src"))

import retrieval.agent_grep as ag  # noqa: E402
from retrieval.agent_grep import RuleCorpus  # noqa: E402


def _rule(rid: str, *, name: str = "", scene: str = "", source: str = "",
          category: str = "", keywords: list[str] | None = None) -> dict:
    return {
        "rule_id": rid,
        "major_category": category,
        "subcategory": "",
        "rule_name": name,
        "visual_retrieval_text": scene,
        "source_quote": source,
        "positive_keywords": keywords or [],
        "visual_checkpoints": [],
    }


class RankedGrepTest(unittest.TestCase):
    def setUp(self) -> None:
        # rule_id order is the corpus order; craft so corpus order and strength
        # order disagree.
        self.rules = [
            # matches "opening" only in the low-weight source clause.
            _rule("R-A", source="per the opening clause of the standard"),
            # matches "opening" in the high-weight rule name AND keywords.
            _rule("R-B", name="unprotected opening guardrail",
                  keywords=["floor opening", "opening cover"]),
            # matches "opening" in the medium-weight scene text.
            _rule("R-C", scene="a large opening in the slab"),
        ]
        self.corpus = RuleCorpus(self.rules)
        self._orig_rank = ag.AGENT_GREP_RANK

    def tearDown(self) -> None:
        ag.AGENT_GREP_RANK = self._orig_rank

    def test_default_is_corpus_order(self) -> None:
        ag.AGENT_GREP_RANK = False
        ids = [h["rule_id"] for h in self.corpus.grep("opening")]
        self.assertEqual(ids, ["R-A", "R-B", "R-C"])

    def test_ranked_orders_by_strength(self) -> None:
        ag.AGENT_GREP_RANK = True
        ids = [h["rule_id"] for h in self.corpus.grep("opening")]
        # R-B (name + 2 keyword hits) strongest, R-C (scene) next, R-A (source) last.
        self.assertEqual(ids, ["R-B", "R-C", "R-A"])

    def test_ranked_preserves_membership_under_cap(self) -> None:
        # When every hit fits under max_hits, ranking only reorders -- the set of
        # returned rule_ids is identical to the default path.
        ag.AGENT_GREP_RANK = False
        default_ids = {h["rule_id"] for h in self.corpus.grep("opening")}
        ag.AGENT_GREP_RANK = True
        ranked_ids = {h["rule_id"] for h in self.corpus.grep("opening")}
        self.assertEqual(default_ids, ranked_ids)

    def test_ranked_truncation_keeps_strongest(self) -> None:
        ag.AGENT_GREP_RANK = True
        top = self.corpus.grep("opening", max_hits=1)
        self.assertEqual([h["rule_id"] for h in top], ["R-B"])

    def test_ties_fall_back_to_corpus_order(self) -> None:
        rules = [
            _rule("R-1", name="opening protection"),
            _rule("R-2", name="opening protection"),
        ]
        corpus = RuleCorpus(rules)
        ag.AGENT_GREP_RANK = True
        ids = [h["rule_id"] for h in corpus.grep("opening")]
        self.assertEqual(ids, ["R-1", "R-2"])


if __name__ == "__main__":
    unittest.main()
