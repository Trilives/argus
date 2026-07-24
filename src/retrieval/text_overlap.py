"""Text-overlap RAG retriever (the CS baseline method).

Dependency-free weighted token overlap between a scene query and each rule's
precomputed ``evidence_chain_text`` in the rule index. This is the baseline the
agentic grep retriever is compared against.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from retrieval.base import RetrievedRule

_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[一-鿿]")


def tokenize_text(text: str) -> Counter[str]:
    """Tokenize mixed CJK/ASCII text into unigrams plus CJK bigrams."""
    tokens: Counter[str] = Counter()
    lowered = text.lower()
    for word in _ASCII_WORD_RE.findall(lowered):
        tokens[word] += 1
    cjk_chars = _CJK_RE.findall(text)
    for char in cjk_chars:
        tokens[char] += 1
    for left, right in zip(cjk_chars, cjk_chars[1:]):
        tokens[left + right] += 2
    return tokens


def weighted_overlap(query: Counter[str], document: Counter[str]) -> float:
    if not query or not document:
        return 0.0
    overlap = sum(min(count, document[token]) for token, count in query.items())
    query_weight = sum(query.values())
    document_weight = sum(document.values())
    return overlap / ((query_weight * document_weight) ** 0.5)


class TextOverlapRetriever:
    """Score a query against a prebuilt rule index by weighted token overlap."""

    method = "text_overlap"

    def __init__(self, rule_index: list[dict[str, Any]], *, text_field: str = "evidence_chain_text") -> None:
        self.rule_index = rule_index
        self.text_field = text_field
        self._doc_tokens = [
            (rule, tokenize_text(str(rule.get(text_field) or ""))) for rule in rule_index
        ]

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedRule]:
        query_tokens = tokenize_text(query)
        scored = [
            (weighted_overlap(query_tokens, doc_tokens), rule)
            for rule, doc_tokens in self._doc_tokens
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
