#!/usr/bin/env python3
"""Model swap (paper-v3): RCASR-BM25 at width 2 run end to end on one model.

Pre-specification: ``docs/paper_v3/design/MODEL_SWAP_PRESPEC.md`` (written before this
script ran). Arms Q9 / Q9F / Q27 replace the scene descriptions, the atom passes and the
judge with one model; REF is the frozen reported configuration. Every method function is
imported unchanged from the frozen analysis (cross-fit, per-fold width-2 re-threshold,
J3-sym judge, E2 metrics, site-unit bootstrap); this file only routes each arm's inputs.

Usage (one model served at a time on the local 4090, commands in Machine.md)::

    uv run --group analysis python experiments/retrieval/eval_model_swap.py --arm q27 --prepare
    RCASR_EXTRACTOR_BASE_URL=http://localhost:8002/v1 \\
        uv run --group analysis python experiments/retrieval/eval_model_swap.py --arm q27 --infer --fit --judge
    ... --arm q27 --latency
    uv run --group analysis python experiments/retrieval/eval_model_swap.py --score
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

CS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CS_ROOT / "src"))
sys.path.insert(0, str(CS_ROOT / "experiments" / "retrieval"))

import config  # noqa: E402
import eval_lowwidth_e2 as lw  # noqa: E402
import rcasr_experiment as rx  # noqa: E402
from atom_posterior import (infer_evidence, pass_specs, prepare_evidence,  # noqa: E402
                            read_evidence, read_pass, request, build_prompt)
from backends import usage  # noqa: E402
from research_snapshot import read_json, sha256, snapshot, verify_snapshot  # noqa: E402
from set_selection import matched_rank_sets  # noqa: E402

PLAN = "docs/paper_v3/design/MODEL_SWAP_PRESPEC.md"
OUT = CS_ROOT / "results/2026-09-24_model_swap"
V2 = "results/2026-09-22_rcasr_v2"
LOWWIDTH = "results/2026-09-22_rcasr_lowwidth"
FROZEN_9B_EVIDENCE = "results/2026-09-22_rcasr_evidence"
ARMS = {"q9": "data/rules/proposed/rcasr_v1_ms_q9.json",
        "q9f": "data/rules/proposed/rcasr_v1_ms_q9f.json",
        "q27": "data/rules/proposed/rcasr_v1_ms_q27.json"}
WIDTHS = {"rcasr_w2": 2.0}
LATENCY_N, SEED = 30, 20260922
DEFAULT_URL = "http://localhost:8001/v1"
FILES = ("experiments/retrieval/eval_model_swap.py", "experiments/retrieval/eval_lowwidth_e2.py", PLAN,
         "src/pipeline.py", "src/prompts.py", "src/schemas.py", "src/symbolic_judgement.py", "src/rules.py",
         "src/backends/openai_api.py", "src/config.py", "Prompts_en/decoupled/rule_evidence_prompt.md",
         "data/rule_assets/rule_units.json", f"{V2}/input_snapshot.json", f"{V2}/rcasr_results.json",
         f"{LOWWIDTH}/lowwidth_results.json", f"{LOWWIDTH}/lowwidth_rows.jsonl", *rx.ANALYSIS_FILES)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def arm_config(arm: str) -> dict:
    return read_json(CS_ROOT / ARMS[arm])


def evidence_dir(arm: str) -> str:
    """Q9 reuses the frozen 9B evidence (same checkpoint, same config, greedy)."""
    return FROZEN_9B_EVIDENCE if arm == "q9" else f"results/2026-09-24_model_swap/{arm}/evidence"


def reference_context() -> dict:
    v2 = read_json(CS_ROOT / V2 / "input_snapshot.json")
    verify_snapshot(CS_ROOT, v2)
    return rx.context(CS_ROOT, v2)


def site_images(ref: dict) -> dict[str, Path]:
    names = {Path(n).stem: n for n in read_json(CS_ROOT / "data/annotations/image_rule_gold.json")["images"]}
    return {i: CS_ROOT / "data/images" / names[i] for i in sorted(ref["split"]["folds"])}


# --- freeze -------------------------------------------------------------------------------

def prepare(arm: str) -> None:
    out = OUT / arm
    if (out / "input_snapshot.json").exists():
        raise FileExistsError("existing freeze cannot be overwritten")
    ref = reference_context()
    if arm != "q9":
        prepare_evidence(CS_ROOT, CS_ROOT / evidence_dir(arm), ARMS[arm], site_images(ref),
                         "346 site-assigned internal images (model-swap arm)")
    ev = CS_ROOT / evidence_dir(arm)
    files = [CS_ROOT / f for f in (*FILES, ARMS[arm])] + [ev / "input_snapshot.json"]
    files += sorted(p for p in (CS_ROOT / "Prompts_en").rglob("*") if p.is_file())
    frozen = snapshot(CS_ROOT, files)
    frozen.update(arm=arm, plan_path=PLAN, config_path=ARMS[arm], evidence_dir=evidence_dir(arm),
                  model=arm_config(arm)["extractor"]["served_model_name"], guided_json=True,
                  created_at=datetime.now(timezone.utc).isoformat(),
                  scope="346 site images; descriptions, atoms and judge on one model; retrospective")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "input_snapshot.json", frozen)


def check(arm: str) -> dict:
    frozen = read_json(OUT / arm / "input_snapshot.json")
    verify_snapshot(CS_ROOT, frozen)
    return frozen


# --- inference: atoms, then scene descriptions ---------------------------------------------

def backend_for(arm: str, url: str):
    from backends.openai_api import OpenAIBackend  # noqa: PLC0415

    config.GUIDED_JSON = True
    return OpenAIBackend(arm_config(arm)["extractor"]["served_model_name"], base_url=url)


def facts_stage(arm: str, url: str, workers: int) -> None:
    from images import load_image  # noqa: PLC0415
    from pipeline import extract_facts  # noqa: PLC0415

    cache, backend = OUT / arm / "facts.jsonl", backend_for(arm, url)
    images = site_images(reference_context())
    done = {r["image_id"] for r in read_jsonl(cache)}
    usage.LEDGER.reset()

    def work(image_id: str) -> dict:
        started = time.perf_counter()
        facts, _, error = extract_facts(backend, load_image(images[image_id]), mode="generic",
                                        retrieval_method="agent_grep")
        return {"image_id": image_id, "image_facts": facts, "parse_error": error,
                "model": backend.model, "seconds": time.perf_counter() - started}

    todo = [i for i in sorted(images) if i not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, future in enumerate(as_completed([pool.submit(work, i) for i in todo]), 1):
            with cache.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            if n % 50 == 0 or n == len(todo):
                print(f"facts {len(done) + n}/{len(images)}", flush=True)
    write_json(OUT / arm / "usage_facts.json", {"images": len(todo), "ledger": usage.LEDGER.snapshot()})


def infer(arm: str, url: str, workers: int) -> None:
    check(arm)
    if arm != "q9":
        infer_evidence(CS_ROOT, CS_ROOT / evidence_dir(arm), url, workers=workers)
    facts_stage(arm, url, workers)


# --- fitting and selection -----------------------------------------------------------------

def bm25_priors(arm: str, ids: list[str]) -> tuple[dict, int]:
    from retrieval.base import facts_query  # noqa: PLC0415
    from retrieval.bm25 import BM25Retriever  # noqa: PLC0415

    index = read_json(CS_ROOT / "data/rule_assets/rule_index.json")
    retriever = BM25Retriever(index)
    rows = {r["image_id"]: r for r in read_jsonl(OUT / arm / "facts.jsonl")}
    if set(rows) != set(ids):
        raise ValueError("scene descriptions are incomplete for the 346 site images")
    priors = {}
    for image_id in ids:
        ranked = retriever.retrieve(facts_query(rows[image_id]["image_facts"] or [""]), top_k=len(index))
        priors[image_id] = {r.rule_id: r.score / (1 + r.score) for r in ranked}
    return priors, sum(bool(r["parse_error"]) or not r["image_facts"] for r in rows.values())


def arm_context(arm: str, ref: dict) -> tuple[dict, int]:
    """The reference context with the arm's atoms and BM25 priors swapped in; folds unchanged."""
    ids = sorted(ref["split"]["folds"])
    ev = CS_ROOT / evidence_dir(arm)
    ev_frozen = read_json(ev / "input_snapshot.json")
    records = read_evidence(ev / "atom_evidence.jsonl", ev_frozen, sha256(ev / "input_snapshot.json"))
    records = {i: records[i] for i in ids}
    priors, failed = bm25_priors(arm, ids)
    ctx = {**ref, "records": records, "posts": rx.merged_posteriors(records),
           "priors": {**ref["priors"], "bm25": priors}}
    ctx["_grids"] = rx.evidence_grids(ctx)
    return ctx, failed


def fit(arm: str) -> None:
    check(arm)
    ref = reference_context()
    ctx, failed = arm_context(arm, ref)
    variant = rx.variants("bm25", ctx["config"])[0]
    run = rx.crossfit(ctx, variant, ctx["_grids"])
    summary = rx.arm_summary(ctx, variant, run)
    w3 = run["selected"]
    sets = {"rcasr_w3": w3, "rcasr_w2": lw.rethreshold(ctx, run, 2.0),
            "bm25_w3": matched_rank_sets(rx.subset(ctx["priors"]["bm25"], sorted(w3)),
                                         sum(map(len, w3.values())))}
    write_json(OUT / arm / "selection.json", sets)
    write_json(OUT / arm / "fit.json", {
        "failed_descriptions": failed, "folds": summary["folds"], "metrics": summary["metrics"],
        "reference_metrics": summary["reference_metrics"],
        "C3_P1_rcasr_bm25_vs_bm25": rx.gate(summary["vs_matched_prior"], ctx["config"]["gates"]["recall_delta"]),
        "widths": {k: sum(map(len, s.values())) / len(s) for k, s in sets.items()}})
    print(json.dumps(read_json(OUT / arm / "fit.json")["widths"], indent=2))


# --- judge ---------------------------------------------------------------------------------

def judge(arm: str, url: str, workers: int) -> None:
    from images import load_image  # noqa: PLC0415

    check(arm)
    sets, backend = read_json(OUT / arm / "selection.json"), backend_for(arm, url)
    facts = {r["image_id"]: r["image_facts"] for r in read_jsonl(OUT / arm / "facts.jsonl")}
    images = site_images(reference_context())
    cache = OUT / arm / "judge_pairs.jsonl"
    done = {(d["image_id"], d["rule_id"]) for d in read_jsonl(cache)}
    pairs = sorted({(i, r) for s in sets.values() for i, rs in s.items() for r in rs} - done)
    usage.LEDGER.reset()

    def work(pair):
        started = time.perf_counter()
        item = rx.judge_pair(backend, load_image(images[pair[0]]), facts[pair[0]] or [], pair[1])
        return pair, {**item, "seconds": time.perf_counter() - started}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, future in enumerate(as_completed([pool.submit(work, p) for p in pairs]), 1):
            (image_id, rule_id), item = future.result()
            with cache.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"image_id": image_id, "rule_id": rule_id, "model": backend.model,
                                         "server": "local-4090", "guided_json": True, **item},
                                        ensure_ascii=False) + "\n")
            if n % 100 == 0 or n == len(pairs):
                print(f"judged {len(done) + n}/{len(done) + len(pairs)}", flush=True)
    write_json(OUT / arm / "usage_judge.json", {"pairs": len(pairs), "ledger": usage.LEDGER.snapshot()})


# --- per-frame latency, one frame at a time as on the robot --------------------------------

def latency(arm: str, url: str, label: str) -> None:
    from openai import OpenAI  # noqa: PLC0415
    from images import encode_image_data_url, load_image  # noqa: PLC0415
    from pipeline import extract_facts  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    check(arm)
    cfg, sets = arm_config(arm), read_json(OUT / arm / "selection.json")
    images = site_images(reference_context())
    sample = sorted(random.Random(SEED).sample(sorted(images), LATENCY_N))
    backend, client, passes = backend_for(arm, url), OpenAI(base_url=url, api_key="EMPTY"), pass_specs(CS_ROOT, cfg)
    rows = []
    for image_id in sample:
        started = time.perf_counter()
        with Image.open(images[image_id]) as handle:
            data_url = encode_image_data_url(handle.convert("RGB"), fmt="PNG")
        image = load_image(images[image_id])

        def atom_pass(name):
            questions, template, max_tokens = passes[name]
            return read_pass(request(client, cfg["extractor"]["served_model_name"],
                                     build_prompt(questions, template), data_url, cfg["decoding"], max_tokens),
                             questions)["status"]

        with ThreadPoolExecutor(max_workers=3) as pool:
            facts_future = pool.submit(extract_facts, backend, image, mode="generic", retrieval_method="agent_grep")
            pass_futures = [pool.submit(atom_pass, n) for n in ("subject", "state")]
            statuses = [f.result() for f in pass_futures]
            facts = facts_future.result()[0]
        perceived = time.perf_counter()
        selected = sets["rcasr_w2"][image_id]
        with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
            labels = [f.result()["label"] for f in [pool.submit(rx.judge_pair, backend, image, facts, r)
                                                     for r in selected]]
        rows.append({"image_id": image_id, "perception_s": perceived - started,
                     "total_s": time.perf_counter() - started, "n_judged": len(selected),
                     "atom_status": statuses, "labels": labels})
    totals = np.array([r["total_s"] for r in rows])
    write_json(OUT / arm / f"latency_{label}.json", {
        "label": label, "n": len(rows), "errors": sum(any(s != "ok" for s in r["atom_status"]) for r in rows),
        "p50_s": float(np.quantile(totals, .5)), "p95_s": float(np.quantile(totals, .95)),
        "perception_p50_s": float(np.median([r["perception_s"] for r in rows])), "rows": rows})
    print(json.dumps({k: v for k, v in read_json(OUT / arm / f"latency_{label}.json").items() if k != "rows"}))


# --- scoring -------------------------------------------------------------------------------

def verdicts_of(path: Path) -> dict:
    return {(d["image_id"], d["rule_id"]): d["label"] for d in read_jsonl(path)}


def cross_arm_ci(units, left, lv, right, rv, statuses, seed, repeats) -> dict:
    """Paired site-unit bootstrap where each arm keeps its own verdicts."""
    rows = []
    for ids in units:
        row = [0.0] * 6
        for i in ids:
            truth = {r for r, s in statuses[i].items() if s == "non_compliant"}
            for col, (sel, ver) in enumerate(((left, lv), (right, rv))):
                pred = {r for r in sel[i] if ver[(i, r)] == "non_compliant"}
                row[col] += len(pred & truth)
                row[col + 3] += len(pred - truth)
            row[2] += len(truth)
        row[5] = len(ids)
        rows.append(row)
    array = np.array(rows, dtype=float)
    draws = array[np.random.default_rng(seed).integers(0, len(array), (repeats, len(array)))].sum(axis=1)
    q = lambda v: [float(x) for x in np.quantile(v, [.025, .975])]
    return {"gv_recall_delta_ci": q((draws[:, 0] - draws[:, 1]) / draws[:, 2]),
            "fa_per_image_delta_ci": q((draws[:, 3] - draws[:, 4]) / draws[:, 5])}


def wording(ci: list[float], gain: str, cost: str) -> str:
    if ci[0] > 0:
        return gain
    if ci[1] < 0:
        return cost
    return "no detectable difference from the reference"


def score() -> None:
    ref = reference_context()
    cfg = ref["config"]["gates"]
    units, statuses = rx.site_units(ref), rx.gold_statuses(CS_ROOT)
    family = {r["rule_id"]: r["rule_id"].split("-")[1] for r in read_json(CS_ROOT / "data/rules/rules_en.json")}
    ref_rows = {r["image_id"]: r for r in read_jsonl(CS_ROOT / LOWWIDTH / "lowwidth_rows.jsonl")}
    ref_sets = {"rcasr_w2": {i: r["rcasr_bm25_w2"] for i, r in ref_rows.items()},
                "bm25_w3": {i: r["bm25_w3"] for i, r in ref_rows.items()}}
    ref_verdicts = lw.load_verdicts(CS_ROOT / "results/2026-09-22_rcasr_judge/internal_pairs.jsonl")
    result = {"plan": PLAN, "reference": {n: rx.e2e_metrics(s, ref_verdicts, statuses, family)
                                          for n, s in ref_sets.items()}, "arms": {}}
    for arm in ARMS:
        if not (OUT / arm / "judge_pairs.jsonl").exists():
            continue
        check(arm)
        sets, verdicts = read_json(OUT / arm / "selection.json"), verdicts_of(OUT / arm / "judge_pairs.jsonl")
        metrics = {n: rx.e2e_metrics(s, verdicts, statuses, family) for n, s in sets.items()}
        c1 = lw.paired_e2e_ci(units, sets["rcasr_w2"], sets["bm25_w3"], verdicts, statuses,
                              cfg["bootstrap_seed"], cfg["bootstrap_repeats"])
        d_r = metrics["rcasr_w2"]["gv_recall"] - metrics["bm25_w3"]["gv_recall"]
        d_f = metrics["rcasr_w2"]["false_alarms_per_image"] - metrics["bm25_w3"]["false_alarms_per_image"]
        c2 = cross_arm_ci(units, sets["rcasr_w2"], verdicts, ref_sets["rcasr_w2"], ref_verdicts, statuses,
                          cfg["bootstrap_seed"], cfg["bootstrap_repeats"])
        judged = read_jsonl(OUT / arm / "judge_pairs.jsonl")
        result["arms"][arm] = {
            "model": arm_config(arm)["extractor"]["served_model_name"], "metrics": metrics,
            "C1_within_arm": {"gv_recall_delta": d_r, "fa_per_image_delta": d_f, **c1,
                              "W1_recall_noninferior": bool(c1["gv_recall_delta_ci"][0] > -lw.MARGIN),
                              "W2_fewer_false_alarms": bool(c1["fa_per_image_delta_ci"][1] < 0)},
            "C2_vs_reference": {
                "gv_recall_delta": metrics["rcasr_w2"]["gv_recall"] - result["reference"]["rcasr_w2"]["gv_recall"],
                "fa_per_image_delta": metrics["rcasr_w2"]["false_alarms_per_image"]
                - result["reference"]["rcasr_w2"]["false_alarms_per_image"], **c2,
                "recall_wording": wording(c2["gv_recall_delta_ci"], "recall gain", "recall cost"),
                "fa_wording": wording(c2["fa_per_image_delta_ci"], "more false alarms", "fewer false alarms")},
            "C3_P1": read_json(OUT / arm / "fit.json")["C3_P1_rcasr_bm25_vs_bm25"],
            "fit": {k: read_json(OUT / arm / "fit.json")[k] for k in ("failed_descriptions", "widths")},
            "judge": {"pairs": len(judged), "parse_errors": sum(bool(d["parse_error"]) for d in judged),
                      "seconds_mean": float(np.mean([d["seconds"] for d in judged]))},
            "latency": {p.stem: {k: v for k, v in read_json(p).items() if k != "rows"}
                        for p in sorted((OUT / arm).glob("latency_*.json"))}}
    write_json(OUT / "model_swap_results.json", result)
    print(json.dumps({a: {"C1": {k: v for k, v in r["C1_within_arm"].items() if k.startswith("W")},
                          "C2": {k: r["C2_vs_reference"][k] for k in ("gv_recall_delta", "gv_recall_delta_ci",
                                                                        "fa_per_image_delta", "fa_per_image_delta_ci")}}
                      for a, r in result["arms"].items()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--arm", choices=tuple(ARMS))
    for flag in ("--prepare", "--infer", "--fit", "--judge", "--latency", "--score"):
        parser.add_argument(flag, action="store_true")
    parser.add_argument("--latency-label", default="gpu48")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    url = os.environ.get("RCASR_EXTRACTOR_BASE_URL", DEFAULT_URL)
    if any((args.prepare, args.infer, args.fit, args.judge, args.latency)) and not args.arm:
        parser.error("--arm is required for per-arm stages")
    if args.prepare:
        prepare(args.arm)
    if args.infer:
        infer(args.arm, url, args.workers)
    if args.fit:
        fit(args.arm)
    if args.judge:
        judge(args.arm, url, args.workers)
    if args.latency:
        latency(args.arm, url, args.latency_label)
    if args.score:
        score()


if __name__ == "__main__":
    main()
