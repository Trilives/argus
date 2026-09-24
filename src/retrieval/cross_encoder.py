"""Cross-encoder reranker over the rule library (off-the-shelf, no training).

With only 42 rules, every rule can be scored jointly with the query, so the
absence of a reranker in the comparison set is conspicuous: BM25 and
``text_overlap`` score query and document independently, and a reviewer will
reasonably ask whether a modern cross-attention scorer closes the gap to the
agentic retrievers without any agent at all. This is that baseline.

It consumes the *same* inputs as R1/BM25 — cached Stage-1 facts as the query,
the rule index's ``evidence_chain_text`` as the document — so a metric
difference is attributable to the scoring function alone. There is no
first-stage retrieval to rerank: all 42 rules are scored, which is the
strongest form of the baseline and removes recall loss from a cascade.

The model is a sequence-classification cross-encoder loaded with plain
``transformers`` (already a core dependency); no ``sentence-transformers``
install is required.
"""

from __future__ import annotations

from typing import Any

from retrieval.base import RetrievedRule

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 512
BATCH_SIZE = 42  # the whole library scores in one forward pass


class CrossEncoderRetriever:
    """Rank all rules by a cross-encoder relevance score for (query, rule text)."""

    method = "cross_encoder"

    def __init__(
        self,
        rule_index: list[dict[str, Any]],
        *,
        model_name: str = DEFAULT_MODEL,
        text_field: str = "evidence_chain_text",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.rule_index = rule_index
        self.text_field = text_field
        self.model_name = model_name
        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device).eval()
        self._docs = [str(rule.get(text_field) or "") for rule in rule_index]

    def _score(self, query: str) -> list[float]:
        torch = self._torch
        scores: list[float] = []
        for start in range(0, len(self._docs), BATCH_SIZE):
            batch = self._docs[start : start + BATCH_SIZE]
            encoded = self._tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**encoded).logits
            # Regression-head cross-encoders emit one logit; a 2-way head emits
            # two, where the positive class is the relevance score.
            column = logits[:, 0] if logits.shape[-1] == 1 else logits[:, 1]
            scores.extend(column.float().cpu().tolist())
        return scores

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedRule]:
        if not query.strip():
            return []
        scores = self._score(query)
        ranked = sorted(zip(scores, self.rule_index), key=lambda pair: -pair[0])[:top_k]
        return [
            RetrievedRule(
                rule_id=rule["rule_id"],
                score=round(float(score), 4),
                rank=rank,
                major_category=rule.get("major_category"),
                subcategory=rule.get("subcategory"),
                rule_name=rule.get("rule_name"),
            )
            for rank, (score, rule) in enumerate(ranked, 1)
        ]
