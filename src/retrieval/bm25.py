"""Okapi BM25 retriever over the prebuilt rule index.

The protocol (EXPERIMENT_PROTOCOL.md §3 RQ3) asks for at least one credible
text-retrieval baseline beyond the simple weighted-overlap scorer; BM25 is
that baseline. It consumes the exact same inputs as ``text_overlap`` (R1):
the same cached Stage-1 facts as the query and the same
``evidence_chain_text`` field of the rule index as the document, tokenized
with the same mixed CJK-bigram/ASCII tokenizer — so any metric difference is
attributable to the scoring function alone.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from retrieval.base import RetrievedRule
from retrieval.text_overlap import tokenize_text

BM25_K1 = 1.5
BM25_B = 0.75


class BM25Retriever:
    """Okapi BM25 (k1/b defaults, smoothed non-negative idf) over the rule index."""

    method = "bm25"

    def __init__(
        self,
        rule_index: list[dict[str, Any]],
        *,
        text_field: str = "evidence_chain_text",
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self.rule_index = rule_index
        self.text_field = text_field
        self.k1 = k1
        self.b = b
        self._docs = [
            (rule, tokenize_text(str(rule.get(text_field) or ""))) for rule in rule_index
        ]
        doc_lengths = [sum(tokens.values()) for _, tokens in self._docs]
        self._doc_lengths = doc_lengths
        self._avgdl = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0
        n_docs = len(self._docs)
        doc_freq: Counter[str] = Counter()
        for _, tokens in self._docs:
            doc_freq.update(tokens.keys())
        # +1 inside the log keeps idf non-negative for very common tokens
        self._idf = {
            token: math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for token, df in doc_freq.items()
        }

    def _score(self, query: Counter[str], doc: Counter[str], doc_len: int) -> float:
        if not query or not doc or not self._avgdl:
            return 0.0
        norm = self.k1 * (1.0 - self.b + self.b * doc_len / self._avgdl)
        score = 0.0
        for token, q_count in query.items():
            tf = doc.get(token, 0)
            if not tf:
                continue
            idf = self._idf.get(token, 0.0)
            score += q_count * idf * (tf * (self.k1 + 1.0)) / (tf + norm)
        return score

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedRule]:
        query_tokens = tokenize_text(query)
        scored = [
            (self._score(query_tokens, doc_tokens, doc_len), rule)
            for (rule, doc_tokens), doc_len in zip(self._docs, self._doc_lengths)
        ]
        ranked = sorted(scored, key=lambda item: (-item[0], item[1].get("rule_id", "")))[:top_k]
        return [
            RetrievedRule(
                rule_id=rule["rule_id"],
                score=score,
                rank=rank,
                major_category=rule.get("major_category"),
                subcategory=rule.get("subcategory"),
                rule_name=rule.get("rule_name"),
            )
            for rank, (score, rule) in enumerate(ranked, 1)
        ]
