#!/usr/bin/env python3
"""RCASR on ConstructionSite-10k released labels (paper-v3 E5b, auxiliary only).

Separate script because every RCASR file is pinned by the internal analysis freeze
(results/2026-09-22_rcasr_v2); this one only imports them. Stages, in order:

    --extract   write the test-split images with >= 1 recorded violation (gitignored)
    --prepare   evidence freeze over those images (same config/prompts as internal)
    --infer     atom passes + Stage-1 facts on the local Qwen3.5-9B server
    --final     fit the final RCASR-BM25 configuration on all 346 internal site images
                and freeze it BEFORE any public scoring
    --select    BM25 k=3 and RCASR-BM25 at the same total width, label-free
    (judge)     J3-sym on selected pairs that hit a mapped provision — run by judge_rcasr.py
    --score     detection over scoreable mapped positives; no precision or F1

Only detection over recorded positives is supported: the release is violation-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys

CS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CS_ROOT / "src"))

import rcasr_experiment as rx  # noqa: E402
from atom_posterior import infer_evidence, prepare_evidence, read_evidence  # noqa: E402
from public_map import image_targets  # noqa: E402
from rcasr import features, fit, score_rows, select_at, width_threshold  # noqa: E402
from research_snapshot import read_json, sha256  # noqa: E402
from set_selection import matched_rank_sets  # noqa: E402

CONFIG_PATH = "data/rules/proposed/rcasr_v1.json"
IMAGE_DIR = CS_ROOT / "data/public_eval/images"
LABELS = CS_ROOT / "data/public_eval/public_labels_test.jsonl"
MAP = CS_ROOT / "data/public_eval/public_map_v1.json"
INTERNAL = "results/2026-09-22_rcasr_v2"
WIDTH = 3.0
EXTRACTOR = "Qwen/Qwen3.5-9B"


def labels() -> dict[str, dict]:
    rows = [json.loads(l) for l in LABELS.read_text().splitlines()]
    return {r["image_id"]: r for r in rows if r["violated_public_rules"]}


def extract() -> None:
    import pyarrow.parquet as pq  # noqa: PLC0415

    wanted = set(labels())
    table = pq.read_table(CS_ROOT / "data/ConstructionSite-10k/test.parquet", columns=["image", "image_id"])
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for image, image_id in zip(table["image"].to_pylist(), table["image_id"].to_pylist()):
        if image_id in wanted:
            (IMAGE_DIR / f"{image_id}.jpg").write_bytes(image["bytes"])
    print(f"extracted {sum((IMAGE_DIR / f'{i}.jpg').exists() for i in wanted)} / {len(wanted)} images")


def facts_stage(out: Path, workers: int) -> None:
    """Stage-1 generic facts (agent_grep guidance, as the internal cache) on the 9B server."""
    from backends.openai_api import OpenAIBackend  # noqa: PLC0415
    from images import load_image  # noqa: PLC0415
    from pipeline import extract_facts  # noqa: PLC0415

    frozen = read_json(out / "input_snapshot.json")
    cache = out / "facts.jsonl"
    done = {json.loads(l)["image_id"] for l in cache.read_text().splitlines()} if cache.exists() else set()
    backend = OpenAIBackend(EXTRACTOR, base_url=os.environ.get("RCASR_EXTRACTOR_BASE_URL",
                                                                "http://localhost:8001/v1"))

    def work(image_id: str) -> dict:
        image = load_image(CS_ROOT / frozen["images"][image_id]["path"])
        facts, _, error = extract_facts(backend, image, mode="generic", retrieval_method="agent_grep")
        return {"image_id": image_id, "image_facts": facts, "parse_error": error, "model": EXTRACTOR}

    todo = [i for i in sorted(frozen["images"]) if i not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, future in enumerate(as_completed([pool.submit(work, i) for i in todo]), 1):
            with cache.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            if n % 50 == 0:
                print(f"facts {len(done) + n}/{len(frozen['images'])}", flush=True)


def final_config(out: Path) -> None:
    """Fit on all 346 internal site images with the pre-specified grid; frozen before scoring."""
    if (out / "final_config.json").exists():
        raise FileExistsError("final configuration already frozen")
    root = CS_ROOT
    ctx = rx.context(root, read_json(root / INTERNAL / "input_snapshot.json"))
    ids = sorted(ctx["split"]["folds"])
    grid = ctx["config"]["grid"]
    pairs = rx.tau_pairs(grid)
    table = {p: {i: features(ctx["compiled"], ctx["posts"][i], p[0], p[1], ctx["opening"]) for i in ids}
             for p in pairs}
    params = fit(rx.subset(ctx["priors"]["bm25"], ids), table, rx.subset(ctx["positives"], ids),
                 WIDTH, {**grid, "tau_pairs": pairs})
    rx.write_json(out / "final_config.json", {
        "fitted_on": "346 internal site images (all folds); pre-specified grid; width 3.0",
        "internal_freeze": INTERNAL, "internal_snapshot_sha256": sha256(root / INTERNAL / "input_snapshot.json"),
        "theta": params["theta"], "tau": params["tau"], "tau_open": params["tau_open"],
        "internal_resubstitution_recall": params["recall"],
        "public_threshold_rule": "global threshold at mean width 3.0 on the public population (label-free)"})


def public_scores(out: Path) -> tuple[dict, dict]:
    from retrieval.base import facts_query  # noqa: PLC0415
    from retrieval.bm25 import BM25Retriever  # noqa: PLC0415

    index = read_json(CS_ROOT / "data/rule_assets/rule_index.json")
    retriever = BM25Retriever(index)
    facts = {json.loads(l)["image_id"]: json.loads(l) for l in (out / "facts.jsonl").read_text().splitlines()}
    bm25 = {}
    for image_id, row in sorted(facts.items()):
        ranked = retriever.retrieve(facts_query(row["image_facts"] or [""]), top_k=len(index))
        bm25[image_id] = {r.rule_id: r.score / (1 + r.score) for r in ranked}
    return bm25, facts


def select(out: Path) -> None:
    frozen = read_json(out / "input_snapshot.json")
    config, final = read_json(CS_ROOT / CONFIG_PATH), read_json(out / "final_config.json")
    records = read_evidence(out / "atom_evidence.jsonl", frozen, sha256(out / "input_snapshot.json"))
    compiled = rx.compiled_overlay(CS_ROOT, config)
    posts = rx.merged_posteriors(records)
    bm25, facts = public_scores(out)
    opening = set(config["abstention"]["opening_atoms"])
    feats = {i: features(compiled, posts[i], final["tau"], final["tau_open"], opening) for i in bm25}
    scores = score_rows(bm25, feats, tuple(final["theta"]))
    rcasr = select_at(scores, width_threshold(scores, WIDTH))
    reference = matched_rank_sets(bm25, sum(map(len, rcasr.values())))
    rows = {i: {"rcasr_bm25": rcasr[i], "bm25_matched": reference[i],
                "facts_parse_error": facts[i]["parse_error"]} for i in sorted(bm25)}
    rx.write_json(out / "public_selection.json", rows)


def mapped_pairs(rows: dict) -> list[tuple[str, str]]:
    mapping, gold = read_json(MAP), labels()
    targets = {i: {t for item in image_targets(mapping, gold[i]) for t in item["targets"]} for i in rows}
    return sorted({(i, r) for i, row in rows.items() for arm in ("rcasr_bm25", "bm25_matched")
                   for r in row[arm] if r in targets[i]})


def wilson(hits: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def detection(items: list[dict], chosen: dict, verdicts: dict | None) -> dict:
    out = {}
    for rule in sorted({i["public_rule"] for i in items}):
        rows = [i for i in items if i["public_rule"] == rule]
        hit = sum(any(t in chosen[i["image_id"]] and (verdicts is None or
                      verdicts.get((i["image_id"], t)) == "non_compliant") for t in i["targets"]) for i in rows)
        out[rule] = {"n": len(rows), "detected": hit, "rate": hit / len(rows), "wilson95": wilson(hit, len(rows))}
    return out


def score(out: Path) -> None:
    mapping, gold = read_json(MAP), labels()
    rows = read_json(out / "public_selection.json")
    judged = CS_ROOT / "results/2026-09-22_rcasr_judge/public_pairs.jsonl"
    verdicts = {(d["image_id"], d["rule_id"]): d["label"]
                for d in map(json.loads, judged.read_text().splitlines())}
    exposed = set(read_json(CS_ROOT / "results/2026-09-19/inventory.json")["public"]["known_exposed_ids"])
    items = [t for i in sorted(rows) for t in image_targets(mapping, gold[i]) if t["status"] == "scoreable"]
    result = {"scope": "ConstructionSite-10k test images with >= 1 recorded violation; auxiliary; "
                       "violation-only labels, so detection over mapped positives is the only endpoint",
              "n_images": len(rows), "n_scoreable_positives": len(items),
              "final_config": read_json(out / "final_config.json"), "arms": {}}
    for arm in ("rcasr_bm25", "bm25_matched"):
        chosen = {i: r[arm] for i, r in rows.items()}
        entry = {"width_mean": sum(map(len, chosen.values())) / len(chosen)}
        for label, subset in (("all", items), ("unexposed", [t for t in items if t["image_id"] not in exposed]),
                              ("known_exposed", [t for t in items if t["image_id"] in exposed])):
            entry[label] = {"retrieval": detection(subset, chosen, None),
                            "end_to_end": detection(subset, chosen, verdicts)}
        derived = [i for i in rows if any(t["status"] == "scoreable" for t in image_targets(mapping, gold[i]))]
        entry["derived_any_violation_detected"] = sum(
            any(verdicts.get((i, r)) == "non_compliant" for r in chosen[i]) for i in derived) / len(derived)
        result["arms"][arm] = entry
    rx.write_json(out / "public_results.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results/2026-09-22_rcasr_public")
    for flag in ("--extract", "--prepare", "--infer", "--final", "--select", "--score"):
        parser.add_argument(flag, action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    out = CS_ROOT / args.out
    if args.extract:
        extract()
    if args.prepare:
        images = {i: IMAGE_DIR / f"{i}.jpg" for i in sorted(labels())}
        prepare_evidence(CS_ROOT, out, CONFIG_PATH, images,
                         "ConstructionSite-10k test images with >= 1 recorded violation (auxiliary)")
    if args.infer:
        infer_evidence(CS_ROOT, out, os.environ.get("RCASR_EXTRACTOR_BASE_URL", "http://localhost:8001/v1"),
                       workers=args.workers)
        facts_stage(out, args.workers)
    if args.final:
        final_config(out)
    if args.select:
        select(out)
    if args.score:
        score(out)


if __name__ == "__main__":
    main()
