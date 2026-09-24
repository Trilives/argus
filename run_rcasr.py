#!/usr/bin/env python3
"""Run RCASR on your own images with the frozen configuration the patrol robot runs.

    uv run python run_rcasr.py examples/sample_images            # local sample photographs
    uv run python run_rcasr.py examples/public_images --threshold public --check-public

Per image, with the functions that produced the paper's numbers:

1. perception: three extractor calls, run concurrently. Stage-1 scene facts come from
   ``pipeline.extract_facts``; the subject and state atom passes come from ``atom_posterior``,
   reading token log-probabilities;
2. prior: BM25 over the facts, mapped to ``b / (1 + b)``;
3. score: ``rcasr.features`` and ``score_rows`` with the frozen weights, then a global
   threshold (``select_at``). An empty set is a legal outcome;
4. judge (optional): ``rcasr_experiment.judge_pair`` (symbolic verdict) on every selected
   provision. A pair routed to the no-subject default abstains as ``need_review``, as on the
   robot.

Endpoints come from the environment and are never written to disk:

    RCASR_EXTRACTOR_URL  OpenAI-compatible server for the extractor (default: OPENAI_BASE_URL).
                         It must return token log-probabilities (vLLM does), serving
                         ``extractor.served_model_name`` from the config (Qwen/Qwen3.5-9B).
    RCASR_JUDGE_URL      server for the judge (default: the extractor URL)
    RCASR_JUDGE_MODEL    judge model name (default: the extractor model, see MODELS.md)
    OPENAI_API_KEY       key, or EMPTY for vLLM

Thresholds: ``runtime`` is the width-2 threshold the robot runs; ``public`` is the width-3
threshold of the public auxiliary evaluation, which reproduces the released
``results/2026-09-22_rcasr_public/public_selection.json`` on all 412 images from the recorded
evidence. A number sets the threshold directly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

RUNTIME_CONFIG = ROOT / "configs/rcasr_runtime_v1.json"
PUBLIC_SELECTION = ROOT / "results/2026-09-22_rcasr_public/public_selection.json"
# Recomputed from the public run's recorded facts and atom evidence with
# eval_rcasr_public.select(); it reproduces the released public selection on 412/412 images.
PUBLIC_THRESHOLD = 0.9502567246313469
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Unguided:
    """Stage-1 facts were frozen without guided JSON (``method.facts.guided_json``), while the
    judge runs guided; this strips the schema from the facts call only, as the robot does."""

    def __init__(self, inner) -> None:
        self.inner, self.model = inner, inner.model

    def complete(self, messages, **kwargs):
        kwargs["json_schema"] = None
        return self.inner.complete(messages, **kwargs)


class Rcasr:
    """The robot engine's per-image path, without its telemetry."""

    def __init__(self, runtime: dict, threshold: float, *, judge: bool) -> None:
        import config as cs_config
        from atom_posterior import pass_specs, validate_state
        from backends.openai_api import OpenAIBackend
        from openai import OpenAI
        from retrieval.bm25 import BM25Retriever
        from typed_schema import compile_schema

        cs_config.GUIDED_JSON = runtime["judge"]["guided_json"]
        frozen = read_json(ROOT / runtime["extractor"]["config"])
        self.compiled = compile_schema(read_json(ROOT / "data/rules/rules_en.json"),
                                       read_json(ROOT / frozen["source_overlay"]))
        validate_state(frozen, self.compiled)
        self.passes = pass_specs(ROOT, frozen)
        index = read_json(ROOT / "data/rule_assets/rule_index.json")
        self.retriever, self.library_size = BM25Retriever(index), len(index)
        self.method, self.extractor = runtime["method"], runtime["extractor"]
        self.no_subject_label = runtime["judge"]["no_subject_default"]["label"]
        self.opening = set(self.method["opening_atoms"])
        self.threshold = threshold

        key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        extractor_url = os.environ.get("RCASR_EXTRACTOR_URL") or os.environ.get("OPENAI_BASE_URL")
        if not extractor_url:
            raise SystemExit("set RCASR_EXTRACTOR_URL (or OPENAI_BASE_URL) to an OpenAI-compatible server")
        served = self.extractor["served_model_name"]
        self.client = OpenAI(base_url=extractor_url, api_key=key, timeout=600, max_retries=1)
        facts = OpenAIBackend(served, base_url=extractor_url, api_key=key)
        self.facts_backend = facts if self.method["facts"]["guided_json"] else Unguided(facts)
        self.judge_backend = None
        if judge:
            self.judge_backend = OpenAIBackend(os.environ.get("RCASR_JUDGE_MODEL") or served,
                                               base_url=os.environ.get("RCASR_JUDGE_URL") or extractor_url,
                                               api_key=key)
        self.pool = ThreadPoolExecutor(max_workers=8)

    def analyse(self, image_id: str, image: Image.Image) -> dict:
        started = time.perf_counter()
        facts, facts_error, posts = self.perceive(image)
        scores = self.score(image_id, facts, posts)
        from rcasr import select_at
        selected = select_at({image_id: scores}, self.threshold)[image_id]
        verdicts = self.judge(image, facts, selected) if self.judge_backend else None
        return {"image_id": image_id, "facts": facts, "facts_error": facts_error,
                "unknown_atoms": sum(v is None for v in posts.values()),
                "threshold": self.threshold, "selected": selected,
                "scores": {r: round(s, 6) for r, s in sorted(scores.items(), key=lambda kv: -kv[1])[:8]},
                "verdicts": verdicts, "decision": decision(selected, verdicts),
                "seconds": round(time.perf_counter() - started, 1)}

    def perceive(self, image: Image.Image):
        from atom_posterior import build_prompt, read_pass, request
        from images import encode_image_data_url, resize_image
        from pipeline import extract_facts

        rgb = image.convert("RGB")
        facts_future = self.pool.submit(extract_facts, self.facts_backend, resize_image(rgb),
                                        mode=self.method["facts"]["mode"],
                                        retrieval_method=self.method["facts"]["retrieval_method"])
        data_url = encode_image_data_url(rgb, fmt="PNG")
        model, decoding = self.extractor["served_model_name"], self.extractor["decoding"]
        futures = {name: self.pool.submit(request, self.client, model, build_prompt(questions, template),
                                          data_url, decoding, max_tokens)
                   for name, (questions, template, max_tokens) in self.passes.items()}
        facts, _, facts_error = facts_future.result()
        posts = {}
        for name in ("subject", "state"):
            posts.update(read_pass(futures[name].result(), self.passes[name][0])["posteriors"])
        return facts, facts_error, posts

    def score(self, image_id: str, facts: list[str], posts: dict) -> dict[str, float]:
        from rcasr import features, score_rows
        from retrieval.base import facts_query

        ranked = self.retriever.retrieve(facts_query(list(facts) or [""]), top_k=self.library_size)
        prior = {r.rule_id: r.score / (1 + r.score) for r in ranked}
        m = self.method
        feats = features(self.compiled, posts, m["tau"], m["tau_open"], self.opening)
        return score_rows({image_id: prior}, {image_id: feats}, tuple(m["theta"]))[image_id]

    def judge(self, image: Image.Image, facts: list[str], selected: list[str]) -> list[dict]:
        from images import resize_image

        judged = resize_image(image.convert("RGB"))
        futures = [(rule, self.pool.submit(self.judge_one, judged, facts, rule)) for rule in selected]
        return [{"rule_id": rule, **future.result()} for rule, future in futures]

    def judge_one(self, image: Image.Image, facts: list[str], rule_id: str) -> dict:
        from rcasr_experiment import judge_pair
        import symbolic_judgement

        try:
            item = judge_pair(self.judge_backend, image, list(facts), rule_id)
        except Exception as exc:  # a failed pair abstains; it never becomes "compliant"
            return {"label": None, "error": f"{type(exc).__name__}: {exc}"}
        route = (symbolic_judgement.symbolic_verdict(item["evidence"])["symbolic"]["route"]
                 if item["evidence"] else None)
        label = self.no_subject_label if route == "no_subject_default" else item["label"]
        return {"label": label, "route": route, "error": item["parse_error"]}


def decision(selected: list[str], verdicts: list[dict] | None) -> str:
    if verdicts is None:
        return "selected" if selected else "empty"
    labels = [v["label"] for v in verdicts]
    if "non_compliant" in labels:
        return "violation"
    if any(label in (None, "need_review") for label in labels):
        return "need_review"
    return "clear" if selected else "clear_empty"


def image_paths(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for target in map(Path, targets):
        found = sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) \
            if target.is_dir() else [target]
        paths.extend(found)
    if not paths:
        raise SystemExit("no images found")
    return paths


def check_public(rows: list[dict]) -> None:
    released = read_json(PUBLIC_SELECTION)
    print("\nAgainst the released public selection (same threshold, same extractor):")
    for row in rows:
        ref = released.get(row["image_id"], {}).get("rcasr_bm25")
        if ref is None:
            print(f"  {row['image_id']}: not in the public evaluation")
            continue
        mine, theirs = set(row["selected"]), set(ref)
        union = mine | theirs
        jaccard = len(mine & theirs) / len(union) if union else 1.0
        print(f"  {row['image_id']}: Jaccard {jaccard:.2f}  released {sorted(theirs)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("images", nargs="+", help="image files or directories")
    parser.add_argument("--threshold", default="runtime",
                        help="'runtime' (width 2, the robot), 'public' (width 3, public run) or a number")
    parser.add_argument("--no-judge", action="store_true", help="stop after selection")
    parser.add_argument("--check-public", action="store_true",
                        help="compare selections with the released ConstructionSite-10k selection")
    parser.add_argument("--out", default="results/demo_rcasr/rcasr_runs.jsonl")
    args = parser.parse_args()

    runtime = read_json(RUNTIME_CONFIG)
    threshold = {"runtime": runtime["method"]["threshold"], "public": PUBLIC_THRESHOLD}.get(args.threshold)
    threshold = float(args.threshold) if threshold is None else threshold
    engine = Rcasr(runtime, threshold, judge=not args.no_judge)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("a", encoding="utf-8") as sink:
        for path in image_paths(args.images):
            with Image.open(path) as image:
                row = engine.analyse(path.stem, image)
            rows.append(row)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            labels = ", ".join(f"{v['rule_id']}={v['label']}" for v in row["verdicts"] or []) or \
                ", ".join(row["selected"]) or "-"
            print(f"{row['image_id']}: {row['decision']:<12} {labels}  ({row['seconds']} s)")
    if args.check_public:
        check_public(rows)
    print(f"\nwrote {len(rows)} record(s) to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
