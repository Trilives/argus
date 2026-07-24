"""Shared types for CS rule retrieval.

Both retrieval methods — text-overlap RAG (:mod:`retrieval.text_overlap`) and the
grep agent (:mod:`retrieval.agent_grep`) — return a ranked list of
:class:`RetrievedRule`, so the pipeline and evaluation treat them identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RetrievedRule:
    rule_id: str
    score: float
    rank: int
    major_category: str | None
    subcategory: str | None
    rule_name: str | None


class Retriever(Protocol):
    """A retriever maps a scene query to ranked candidate rules."""

    method: str

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedRule]: ...


def chain_sample_query(sample: dict[str, Any]) -> str:
    """Build the scene query for a curated evidence-chain sample.

    Concatenates the human description, summary, and image facts — the same
    text both retrieval methods receive, so comparisons stay fair.
    """
    facts = sample.get("image_facts") or []
    if not isinstance(facts, list):
        facts = [str(facts)]
    parts = [
        str(sample.get("image_description") or ""),
        str(sample.get("evidence_chain_summary") or ""),
        "。".join(str(fact) for fact in facts),
    ]
    return "。".join(part for part in parts if part)


def facts_query(image_facts: list[str]) -> str:
    """Build a scene query from a list of model-extracted image facts."""
    return "。".join(str(fact) for fact in image_facts if str(fact).strip())
