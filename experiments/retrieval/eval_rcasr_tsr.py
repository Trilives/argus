#!/usr/bin/env python3
"""Two-stage retrieval prior (Todo 3-B): RCASR on a SigLIP-2 -> cross-encoder prior.

Pre-specification: ``docs/paper_v3/design/TWO_STAGE_PRIOR_PRESPEC.md`` (written before this
script ran). The retriever is label-free (``src/two_stage_prior.py``). The analysis imports every
method function from the frozen v2 analysis unchanged and swaps only the prior.

Usage (local GPU, no network)::

    uv run --group analysis python experiments/retrieval/eval_rcasr_tsr.py --prepare
    uv run --group analysis python experiments/retrieval/eval_rcasr_tsr.py --infer
    uv run --group analysis python experiments/retrieval/eval_rcasr_tsr.py --evaluate
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

CS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CS_ROOT / "src"))

import rcasr_experiment as rx  # noqa: E402
from research_snapshot import read_json, sha256, snapshot, verify_snapshot  # noqa: E402
from set_selection_metrics import metrics  # noqa: E402
from two_stage_prior import pin, rank_prior  # noqa: E402

CONFIG = "data/rules/proposed/rcasr_v1_tsr.json"
PLAN = "docs/paper_v3/design/TWO_STAGE_PRIOR_PRESPEC.md"
OUT = CS_ROOT / "results/2026-09-27_rcasr_tsr"
V2 = "results/2026-09-22_rcasr_v2"
FILES = (CONFIG, PLAN, "experiments/retrieval/eval_rcasr_tsr.py", "src/two_stage_prior.py",
         "src/retrieval/siglip.py", "src/retrieval/cross_encoder.py", "src/images.py", "src/config.py",
         "tests/test_two_stage_prior.py", f"{V2}/input_snapshot.json", f"{V2}/rcasr_rows.jsonl",
         f"{V2}/rcasr_results.json", *rx.ANALYSIS_FILES)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def gold_images() -> dict[str, Path]:
    names = read_json(CS_ROOT / "data/annotations/image_rule_gold.json")["images"]
    return {Path(n).stem: CS_ROOT / "data/images" / n for n in sorted(names)}


def parameters(model_id: str, revision: str) -> int:
    from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    path = try_to_load_from_cache(model_id, "model.safetensors", revision=revision)
    total = 0
    with safe_open(path, framework="pt") as handle:
        for key in handle.keys():
            n = 1
            for d in handle.get_slice(key).get_shape():
                n *= d
            total += n
    return total


def model_pins(config: dict) -> dict:
    return {stage: {"model_id": config[stage]["model_id"], "revision": config[stage]["revision"],
                    "sha256": pin(config[stage]["model_id"], config[stage]["revision"])}
            for stage in ("stage1", "stage2")}


# --- freeze -------------------------------------------------------------------------------

def prepare() -> None:
    if (OUT / "input_snapshot.json").exists():
        raise FileExistsError("existing freeze cannot be overwritten; use a new output directory")
    config = read_json(CS_ROOT / CONFIG)
    files = [CS_ROOT / f for f in FILES] + list(gold_images().values())
    frozen = snapshot(CS_ROOT, files)
    pins = model_pins(config)
    for stage, item in pins.items():
        item["parameters"] = parameters(item["model_id"], item["revision"])
    frozen.update(plan_path=PLAN, config_path=CONFIG, base_freeze=V2, models=pins,
                  created_at=datetime.now(timezone.utc).isoformat(),
                  scope="346 site images, site-grouped cross-fitting on the v2 folds; retrieval level; "
                        "retrospective, not an independent test")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "input_snapshot.json", frozen)
    print(f"froze {len(frozen['file_sha256'])} files -> {OUT / 'input_snapshot.json'}")


def check() -> dict:
    frozen = read_json(OUT / "input_snapshot.json")
    verify_snapshot(CS_ROOT, frozen)
    for stage, item in model_pins(read_json(CS_ROOT / frozen["config_path"])).items():
        if item["sha256"] != frozen["models"][stage]["sha256"]:
            raise ValueError(f"{stage} weights differ from the freeze")
    return frozen


# --- inference (label-free) ---------------------------------------------------------------

def infer() -> None:
    import torch  # noqa: PLC0415

    from images import load_image  # noqa: PLC0415
    from retrieval.base import facts_query  # noqa: PLC0415
    from two_stage_prior import TwoStageRetriever  # noqa: PLC0415

    check()
    target = OUT / "tsr_ranked.json"
    if target.exists():
        raise FileExistsError(f"{target} exists; inference runs once")
    config = read_json(CS_ROOT / CONFIG)
    index = read_json(CS_ROOT / config["library"])
    facts = read_json(CS_ROOT / config["stage2"]["query_cache"])
    images = gold_images()
    started = time.perf_counter()
    retriever = TwoStageRetriever(config, index)
    first = sorted(images)[0]
    retriever.rank(load_image(images[first]), facts_query(facts[first]["image_facts"]))  # warm-up, discarded
    load_seconds = time.perf_counter() - started
    rows = {}
    for n, image_id in enumerate(sorted(images), 1):
        rows[image_id] = retriever.rank(load_image(images[image_id]), facts_query(facts[image_id]["image_facts"]))
        if n % 100 == 0:
            print(f"ranked {n}/{len(images)}", flush=True)
    write_json(target, {"config_path": CONFIG, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                        else "cpu", "torch": torch.__version__, "load_and_warmup_seconds": load_seconds,
                        "created_at": datetime.now(timezone.utc).isoformat(), "rows": rows})
    print(f"wrote {len(rows)} rankings -> {target}")


# --- analysis (reads labels; no model call) -----------------------------------------------

def reference_context() -> dict:
    v2 = read_json(CS_ROOT / V2 / "input_snapshot.json")
    verify_snapshot(CS_ROOT, v2)
    return rx.context(CS_ROOT, v2)


def tsr_context(ref: dict, ranked: dict, depth: int) -> dict:
    """The v2 context with the TSR prior under the ``bm25`` key.

    ``rx.budget`` returns w* = 3 for that key only, which is the pre-specified TSR budget (its
    top-3 is always full); every other function reads the prior through the variant's key.
    """
    priors = {i: rank_prior(ranked[i], ref["library"], depth) for i in ref["priors"]["bm25"]}
    return {**ref, "priors": {"bm25": priors}}


def renamed(variant: rx.Variant) -> rx.Variant:
    return replace(variant, name=variant.name.replace("_bm25", "_tsr"))


def context_comparisons(ctx: dict, ref: dict, sets: dict, held: list[str]) -> dict:
    cfg = ctx["config"]["gates"]
    pos = {i: ctx["positives"][i] for i in held}
    pairs = {"tsr_top3_vs_bm25_top3": ("tsr_top3", "bm25_top3"),
             "rcasr_tsr_vs_rcasr_bm25": ("rcasr_tsr", "rcasr_bm25"),
             "tsr_top3_vs_stage1_top3": ("tsr_top3", "stage1_top3")}
    out = {"metrics": {n: metrics(rx.subset(s, held), pos, ctx["family"]) for n, s in sets.items()}}
    for name, (left, right) in pairs.items():
        out[name] = rx.compare(ctx, rx.subset(sets[left], held), rx.subset(sets[right], held), cfg)
    out["r4_top3_context"] = {"metrics": metrics(rx.subset(ref["r4_top3"], held), pos, ctx["family"]),
                              "note": "R4 at its own width; context only, not a matched comparison"}
    return out


def cost(tsr: dict) -> dict:
    rows = tsr["rows"].values()
    return {"device": tsr["device"], "load_and_warmup_seconds": tsr["load_and_warmup_seconds"],
            **{f"{s}_seconds_per_image": sum(r["seconds"][s] for r in rows) / len(rows)
               for s in ("stage1", "stage2")}}


def evaluate() -> None:
    frozen = check()
    tsr = read_json(OUT / "tsr_ranked.json")
    depth = read_json(CS_ROOT / CONFIG)["stage1"]["depth"]
    ranked = {i: r["ranked"] for i, r in tsr["rows"].items()}
    ref = reference_context()
    ctx = tsr_context(ref, ranked, depth)
    grids = rx.evidence_grids(ctx)
    runs = {}
    for variant in rx.variants("bm25", ctx["config"]):
        run = rx.crossfit(ctx, variant, grids)
        run["summary"] = rx.arm_summary(ctx, variant, run)
        runs[renamed(variant).name] = run
    held = sorted(runs["rcasr_tsr"]["selected"])
    cfg, bar = ctx["config"]["gates"], ctx["config"]["gates"]["recall_delta"]
    sel = lambda name: runs[name]["selected"]
    gates = {"T1_rcasr_tsr_vs_tsr": rx.gate(runs["rcasr_tsr"]["summary"]["vs_matched_prior"], bar),
             "T2_structure_beyond_adaptivity": rx.gate(rx.compare(ctx, sel("rcasr_tsr"), sel("adaptive_tsr"), cfg), None),
             "TN_null_tsr": rx.gate(runs["null_tsr"]["summary"]["vs_matched_prior"], bar)}
    v2_rows = {json.loads(l)["image_id"]: json.loads(l) for l in (CS_ROOT / V2 / "rcasr_rows.jsonl").read_text().splitlines()}
    reference = rx.matched_reference(ctx, rx.Variant("ref", "bm25"), sel("rcasr_tsr"))
    sets = {"rcasr_tsr": sel("rcasr_tsr"), "tsr_top3": {i: ranked[i][:3] for i in held},
            "stage1_top3": {i: [h["rule_id"] for h in tsr["rows"][i]["stage1"][:3]] for i in held},
            "bm25_top3": rx.matched_rank_sets(rx.subset(ref["priors"]["bm25"], held), 3 * len(held)),
            "rcasr_bm25": {i: v2_rows[i]["selected"]["rcasr_bm25"] for i in held}}
    alphas = ctx["config"]["selection"]["alphas"]
    result = {
        "plan": PLAN, "scope": frozen["scope"], "snapshot_sha256": sha256(OUT / "input_snapshot.json"),
        "population": ctx["split"]["counts"], "gates": gates,
        "arms": {name: run["summary"] for name, run in runs.items()},
        "context_width3": context_comparisons(ctx, ref, sets, held),
        "calibration": {f"tsr_{'group' if g else 'pooled'}_{a}": rx.calibration_arm(ctx, rx.Variant("tsr", "bm25"), grids, a, g)
                        for a in alphas for g in (False, True)},
        "per_site": rx.per_site(ctx, {"rcasr_tsr": sel("rcasr_tsr"), "tsr_matched": reference}),
        "family_holdout": rx.family_holdout(ctx, "bm25", grids),
        "harm": rx.harm(ctx, sel("rcasr_tsr"), reference),
        "cost": cost(tsr), "models": frozen["models"],
        "limits": ["all images development-exposed; retrospective cross-fitting, not an independent test",
                   "retrieval level only; no end-to-end judge run (pre-specified)",
                   "the cross-encoder alone was scored on all 500 images before this freeze"]}
    write_json(OUT / "tsr_results.json", result)
    with (OUT / "tsr_rows.jsonl").open("w", encoding="utf-8") as handle:
        for i in held:
            handle.write(json.dumps({"image_id": i, "fold": ctx["split"]["folds"][i], "ranked": ranked[i],
                                     "selected": {n: r["selected"][i] for n, r in runs.items()},
                                     "reference": reference[i]}, ensure_ascii=False) + "\n")
    print(json.dumps(gates, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    for flag in ("--prepare", "--infer", "--evaluate"):
        parser.add_argument(flag, action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.infer:
        infer()
    if args.evaluate:
        evaluate()


if __name__ == "__main__":
    main()
