"""Tests for the BM25 rule retriever (credible text baseline, protocol RQ3)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CS_ROOT / "src"))

from retrieval.bm25 import BM25Retriever  # noqa: E402

INDEX = [
    {"rule_id": "R-A", "rule_name": "安全帽", "evidence_chain_text": "工人未佩戴安全帽 头部防护缺失"},
    {"rule_id": "R-B", "rule_name": "临边", "evidence_chain_text": "临边防护栏杆缺失 坠落风险"},
    {"rule_id": "R-C", "rule_name": "通用", "evidence_chain_text": "施工现场 施工现场 施工现场 工人 作业"},
]


class BM25RetrieverTest(unittest.TestCase):
    def test_topical_match_outranks_generic_document(self):
        retriever = BM25Retriever(INDEX)

        results = retriever.retrieve("现场有工人未佩戴安全帽", top_k=3)

        self.assertEqual(results[0].rule_id, "R-A")
        self.assertGreater(results[0].score, results[1].score)

    def test_respects_top_k_and_ranks_sequentially(self):
        retriever = BM25Retriever(INDEX)

        results = retriever.retrieve("临边防护", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual([r.rank for r in results], [1, 2])
        self.assertEqual(results[0].rule_id, "R-B")

    def test_deterministic_tie_break_by_rule_id(self):
        index = [
            {"rule_id": "R-2", "evidence_chain_text": "完全相同的文本"},
            {"rule_id": "R-1", "evidence_chain_text": "完全相同的文本"},
        ]
        retriever = BM25Retriever(index)

        results = retriever.retrieve("完全相同的文本", top_k=2)

        self.assertEqual([r.rule_id for r in results], ["R-1", "R-2"])

    def test_empty_query_scores_zero(self):
        retriever = BM25Retriever(INDEX)

        results = retriever.retrieve("", top_k=3)

        self.assertTrue(all(r.score == 0.0 for r in results))


if __name__ == "__main__":
    unittest.main()
