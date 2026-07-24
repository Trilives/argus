"""SigLIP-2 image->rule retriever (simulates a CLIP-style retrieval baseline).

Unlike the text-query retrievers (text_overlap / agent_grep), SigLIP consumes the
**image** directly: it embeds the image with SigLIP-2 and ranks rules by cosine
similarity against each rule's ``visual_retrieval_text`` embedding. Two variants —
``224`` and ``512`` input resolution — are selected by model id (see
``config.SIGLIP_MODELS``); the code path is identical.

Heavy imports (torch / transformers) are deferred to first use so the module can be
imported for dry structural checks without loading the model.
"""

from __future__ import annotations

from typing import Any

from retrieval.base import RetrievedRule


def _pooled(out: Any) -> Any:
    """Unwrap SigLIP feature output to the pooled embedding tensor.

    transformers >=5 returns ``BaseModelOutputWithPooling`` from
    ``get_text_features`` / ``get_image_features``; older versions returned the
    pooled tensor directly. Accept both.
    """
    import torch

    if isinstance(out, torch.Tensor):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    raise TypeError(f"Unexpected SigLIP feature output: {type(out).__name__}")


class SiglipRetriever:
    """Rank rules by SigLIP-2 image<->rule-text cosine similarity."""

    method = "siglip"

    def __init__(
        self,
        model_id: str,
        rule_units: list[dict[str, Any]],
        *,
        text_field: str = "visual_retrieval_text",
        variant: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.variant = variant or model_id
        self.method = f"siglip_{variant}" if variant else "siglip"
        self.text_field = text_field
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._rule_embs: Any = None
        self._rule_ids = [r["rule_id"] for r in rule_units]
        self._rule_meta = {r["rule_id"]: r for r in rule_units}
        self._rule_texts = [self._text_for(r) for r in rule_units]

    def _text_for(self, rule: dict[str, Any]) -> str:
        text = str(rule.get(self.text_field) or "").strip()
        if not text:  # fall back to a scene-ish description if the field is empty
            text = f"{rule.get('subcategory', '')} {rule.get('rule_name', '')}".strip()
        return text

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModel.from_pretrained(self.model_id)
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(device).eval()
        self._device = device
        with torch.no_grad():
            inputs = self._processor(
                text=self._rule_texts, padding="max_length", max_length=64,
                truncation=True, return_tensors="pt",
            ).to(device)
            emb = _pooled(self._model.get_text_features(**inputs))
            self._rule_embs = emb / emb.norm(dim=-1, keepdim=True)

    def retrieve_image(self, image: Any, *, top_k: int = 5) -> list[RetrievedRule]:
        """Rank rules for one image (the SigLIP retrieval query is the image itself)."""
        import torch

        self._ensure_loaded()
        with torch.no_grad():
            inputs = self._processor(images=[image], return_tensors="pt").to(self._device)
            img = _pooled(self._model.get_image_features(**inputs))
            img = img / img.norm(dim=-1, keepdim=True)
            sims = (img @ self._rule_embs.T).squeeze(0)
            k = min(top_k, sims.shape[0])
            vals, idx = torch.topk(sims, k)

        results = []
        for rank, (i, score) in enumerate(zip(idx.tolist(), vals.tolist()), start=1):
            rid = self._rule_ids[i]
            meta = self._rule_meta[rid]
            results.append(
                RetrievedRule(
                    rule_id=rid,
                    score=float(score),
                    rank=rank,
                    major_category=meta.get("major_category"),
                    subcategory=meta.get("subcategory"),
                    rule_name=meta.get("rule_name"),
                )
            )
        return results

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedRule]:  # pragma: no cover
        raise NotImplementedError(
            "SiglipRetriever is image-based; call retrieve_image(image, top_k=...) instead."
        )
