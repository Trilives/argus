"""Two-stage retrieval prior for RCASR (Todo 3-B): vector retrieval, then cross-encoder rerank.

Stage 1 ranks the library by SigLIP-2 image/provision-text cosine and keeps the top M.
Stage 2 reorders those M by a cross-encoder over (scene description, provision text).
The reranked list becomes a prior with the R4 rank map written for a list of length M.
Both stages reuse the unchanged retrievers in ``retrieval``; nothing here reads a label.
Pre-specification: ``docs/paper_v3/design/TWO_STAGE_PRIOR_PRESPEC.md``.
"""
from __future__ import annotations

from pathlib import Path
import time

from research_snapshot import sha256


def cascade(stage1: list[str], rerank: dict[str, float]) -> list[str]:
    """Stage-1 candidates reordered by rerank score; ties keep stage-1 order."""
    return [rule for _, _, rule in sorted((-rerank[rule], i, rule) for i, rule in enumerate(stage1))]


def rank_prior(ranked: list[str], library: list[str], length: int) -> dict[str, float]:
    """(L+1-rank)/(L+1) for listed provisions, 0.5/(L+1) otherwise; L = 4 is the R4 map."""
    if len(ranked) > length:
        raise ValueError(f"list of {len(ranked)} exceeds its declared length {length}")
    rank = {rule: i + 1 for i, rule in enumerate(ranked)}
    return {rule: (length + 1 - rank[rule]) / (length + 1) if rule in rank else .5 / (length + 1)
            for rule in library}


def pin(model_id: str, revision: str, cache: Path | None = None) -> dict[str, str]:
    """SHA-256 of every file in the cached snapshot; the cache's ``main`` must be ``revision``."""
    if cache is None:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415
        cache = Path(HF_HUB_CACHE)
    repo = Path(cache) / f"models--{model_id.replace('/', '--')}"
    main = (repo / "refs" / "main").read_text().strip()
    if main != revision:
        raise ValueError(f"{model_id}: cached revision {main} is not the frozen {revision}")
    snap = repo / "snapshots" / revision
    return {p.name: sha256(p) for p in sorted(snap.iterdir()) if p.is_file()}


class TwoStageRetriever:
    """SigLIP-2 top-M, reranked by a cross-encoder; one image at a time."""

    def __init__(self, config: dict, index: list[dict], device: str | None = None) -> None:
        from retrieval.cross_encoder import CrossEncoderRetriever  # noqa: PLC0415
        from retrieval.siglip import SiglipRetriever  # noqa: PLC0415

        s1, s2 = config["stage1"], config["stage2"]
        self.depth = s1["depth"]
        self.library = [r["rule_id"] for r in index]
        self.stage1 = SiglipRetriever(s1["model_id"], index, text_field=s1["text_field"], device=device)
        self.stage2 = CrossEncoderRetriever(index, model_name=s2["model_id"], text_field=s2["doc_field"],
                                            device=device)

    def rank(self, image, query: str) -> dict:
        started = time.perf_counter()
        hits = self.stage1.retrieve_image(image, top_k=self.depth)
        middle = time.perf_counter()
        scores = {r.rule_id: r.score for r in self.stage2.retrieve(query, top_k=len(self.library))}
        stage1 = [r.rule_id for r in hits]
        ranked = cascade(stage1, scores)
        return {"stage1": [{"rule_id": r.rule_id, "cosine": r.score} for r in hits],
                "rerank_scores": {rule: scores[rule] for rule in stage1}, "ranked": ranked,
                "seconds": {"stage1": middle - started, "stage2": time.perf_counter() - middle}}
