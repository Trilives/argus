#!/usr/bin/env python3
"""Low-width E2 (paper-v3 §E2, R10): RCASR-BM25 at target mean width 2 against BM25 width 3.

Pre-specification: ``docs/paper_v3/design/LOWWIDTH_E2_PRESPEC.md`` (written before this
script ran). The fitted RCASR parameters of the v2 analysis freeze are reused unchanged; only
the per-fold global threshold is re-set on the fitting folds at the lower budget. Every
selected pair is expected to sit inside the width-3 selection and therefore to carry a frozen
verdict from ``results/2026-09-22_rcasr_judge/internal_pairs.jsonl``; the script asserts
the subset relation and judges any missing pair with the same frozen judge, unchanged.

Usage::

    uv run --group analysis python experiments/retrieval/eval_lowwidth_e2.py --prepare
    uv run --group analysis python experiments/retrieval/eval_lowwidth_e2.py --run
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

CS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CS_ROOT / "src"))
sys.path.insert(0, str(CS_ROOT / "experiments" / "retrieval"))

import rcasr_experiment as rx  # noqa: E402
from rcasr import select_at, width_threshold  # noqa: E402
from research_snapshot import read_json, sha256, snapshot, verify_snapshot  # noqa: E402
from set_selection import matched_rank_sets  # noqa: E402

OUT = CS_ROOT / "results/2026-09-22_rcasr_lowwidth"
PLAN = "docs/paper_v3/design/LOWWIDTH_E2_PRESPEC.md"
V2 = "results/2026-09-22_rcasr_v2"
JUDGE = "results/2026-09-22_rcasr_judge"
CONFIG_PATH = "data/rules/proposed/rcasr_v1.json"
TARGET_WIDTH = 2.0
MARGIN = 0.03
DIAGNOSTIC_WIDTHS = (1.0, 1.5, 2.0, 2.5, 3.0)
FILES = ("experiments/retrieval/eval_lowwidth_e2.py", "experiments/retrieval/judge_rcasr.py",
         PLAN, CONFIG_PATH, f"{V2}/input_snapshot.json", f"{V2}/rcasr_rows.jsonl",
         f"{V2}/rcasr_results.json", f"{JUDGE}/input_snapshot.json", f"{JUDGE}/internal_pairs.jsonl",
         "docs/paper_v3/manuscript/figures/f5_data.json", *rx.ANALYSIS_FILES)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def prepare() -> None:
    if (OUT / "input_snapshot.json").exists():
        raise FileExistsError("existing freeze cannot be overwritten")
    frozen = snapshot(CS_ROOT, [CS_ROOT / f for f in FILES])
    frozen.update(plan_path=PLAN, target_width=TARGET_WIDTH, margin=MARGIN,
                  created_at=datetime.now(timezone.utc).isoformat(),
                  scope="346 site images; fold-fitted RCASR-BM25 rethresholded at mean width 2; "
                        "frozen width-3 judge verdicts; retrospective, not an independent test")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "input_snapshot.json", frozen)


# --- selection at the lower budget ---------------------------------------------------------

def rethreshold(ctx: dict, run: dict, width: float) -> dict[str, list[str]]:
    """Fold-fitted scores, threshold re-set on the fitting folds at ``width``."""
    folds, k = ctx["split"]["folds"], ctx["split"]["k"]
    grids = ctx["_grids"]
    variant = rx.variants("bm25", ctx["config"])[0]
    chosen = {}
    for fold in run["folds"]:
        held = fold["held_out"]
        fitting, _, test = rx.fold_roles(folds, k, held)
        params = {"theta": tuple(fold["theta"]), "tau": fold["tau"], "tau_open": fold["tau_open"]}
        fit_scores = rx.scored(ctx, variant, grids, params, fitting)
        threshold = width_threshold(fit_scores, width)
        chosen.update(select_at(rx.scored(ctx, variant, grids, params, test), threshold))
    return chosen


def assert_nested(inner: dict, outer: dict) -> int:
    extra = {(i, r) for i, rs in inner.items() for r in rs if r not in outer[i]}
    if extra:
        raise AssertionError(f"{len(extra)} low-width pairs fall outside the width-3 selection")
    return sum(map(len, inner.values()))


# --- paired site-unit bootstrap on end-to-end statistics ----------------------------------

def unit_rows(units: list[list[str]], left: dict, right: dict, verdicts: dict, statuses: dict) -> np.ndarray:
    rows = []
    for ids in units:
        row = [0.0] * 6
        for i in ids:
            truth = {r for r, s in statuses[i].items() if s == "non_compliant"}
            for col, sel in ((0, left), (1, right)):
                pred = {r for r in sel[i] if verdicts[(i, r)] == "non_compliant"}
                row[col] += len(pred & truth)
                row[col + 3] += len(pred - truth)
            row[2] += len(truth)
        row[5] = len(ids)
        rows.append(row)
    return np.array(rows, dtype=float)


def paired_e2e_ci(units: list[list[str]], left: dict, right: dict, verdicts: dict, statuses: dict,
                  seed: int, repeats: int) -> dict:
    array = unit_rows(units, left, right, verdicts, statuses)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), (repeats, len(array)))].sum(axis=1)
    recall_delta = (draws[:, 0] - draws[:, 1]) / draws[:, 2]
    fa_delta = (draws[:, 3] - draws[:, 4]) / draws[:, 5]
    q = lambda v: [float(x) for x in np.quantile(v, [.025, .975])]
    return {"gv_recall_delta_ci": q(recall_delta), "fa_per_image_delta_ci": q(fa_delta),
            "unit": "confirmed site unit (near-duplicate clusters fused)", "n_units": len(units),
            "repeats": repeats, "seed": seed,
            "interpretation": "site-clustered paired bootstrap over cross-fitted held-out selections; retrospective"}


# --- run ----------------------------------------------------------------------------------

def load_verdicts(cache: Path) -> dict:
    rows = [json.loads(l) for l in cache.read_text().splitlines()]
    return {(d["image_id"], d["rule_id"]): d["label"] for d in rows}


def judge_missing(pairs: list[tuple[str, str]]) -> None:
    import judge_rcasr  # noqa: PLC0415
    from backends.openai_api import OpenAIBackend  # noqa: PLC0415
    from images import load_image  # noqa: PLC0415
    import config  # noqa: PLC0415

    config.GUIDED_JSON = True
    model = read_json(CS_ROOT / CONFIG_PATH)["judge"]["model_id"]
    _, paths, facts = judge_rcasr.internal_work()
    backend = OpenAIBackend(model)
    with (OUT / "extra_pairs.jsonl").open("a", encoding="utf-8") as handle:
        for image_id, rule_id in pairs:
            item = rx.judge_pair(backend, load_image(paths[image_id]), facts[image_id] or [], rule_id)
            handle.write(json.dumps({"image_id": image_id, "rule_id": rule_id, "model": model,
                                     "guided_json": True, **item}, ensure_ascii=False) + "\n")


def run() -> dict:
    frozen = read_json(OUT / "input_snapshot.json")
    verify_snapshot(CS_ROOT, frozen)
    v2_frozen = read_json(CS_ROOT / V2 / "input_snapshot.json")
    verify_snapshot(CS_ROOT, v2_frozen)
    ctx = rx.context(CS_ROOT, v2_frozen)
    ctx["_grids"] = rx.evidence_grids(ctx)
    run_w3 = {"folds": read_json(CS_ROOT / V2 / "rcasr_results.json")["arms"]["rcasr_bm25"]["folds"]}
    rows = {json.loads(l)["image_id"]: json.loads(l) for l in (CS_ROOT / V2 / "rcasr_rows.jsonl").read_text().splitlines()}
    w3_sets = rx.e2_sets(rows)
    sets = {"rcasr_bm25_w2": rethreshold(ctx, run_w3, TARGET_WIDTH),
            "bm25_w3": w3_sets["bm25_matched"], "rcasr_bm25_w3": w3_sets["rcasr_bm25"]}
    if set(sets["rcasr_bm25_w2"]) != set(sets["bm25_w3"]):
        raise ValueError("populations differ")
    total = assert_nested(sets["rcasr_bm25_w2"], sets["rcasr_bm25_w3"])
    sets["bm25_w2_matched"] = matched_rank_sets(rx.subset(ctx["priors"]["bm25"], sorted(sets["bm25_w3"])), total)
    diagnostic = {str(w): rethreshold(ctx, run_w3, w) for w in DIAGNOSTIC_WIDTHS}

    verdicts = load_verdicts(CS_ROOT / JUDGE / "internal_pairs.jsonl")
    needed = {(i, r) for s in [*sets.values(), *diagnostic.values()] for i, rs in s.items() for r in rs}
    missing = sorted(needed - set(verdicts))
    if missing:
        judge_missing(missing)
    if (OUT / "extra_pairs.jsonl").exists():
        verdicts.update(load_verdicts(OUT / "extra_pairs.jsonl"))
    still = sorted(needed - set(verdicts))
    if still:
        raise ValueError(f"{len(still)} pairs still unjudged")

    statuses = rx.gold_statuses(CS_ROOT)
    family = {r["rule_id"]: r["rule_id"].split("-")[1] for r in read_json(CS_ROOT / "data/rules/rules_en.json")}
    units = rx.site_units(ctx)
    cfg = ctx["config"]["gates"]
    arms = {name: rx.e2e_metrics(s, verdicts, statuses, family) for name, s in sets.items()}
    for name in ("rcasr_bm25_w2", "rcasr_bm25_w3", "bm25_w2_matched"):
        arms[name]["vs_bm25_w3"] = {
            "gv_recall_delta": arms[name]["gv_recall"] - arms["bm25_w3"]["gv_recall"],
            "fa_per_image_delta": arms[name]["false_alarms_per_image"] - arms["bm25_w3"]["false_alarms_per_image"],
            **paired_e2e_ci(units, sets[name], sets["bm25_w3"], verdicts, statuses,
                            cfg["bootstrap_seed"], cfg["bootstrap_repeats"])}
    arm = arms["rcasr_bm25_w2"]
    widths = np.array([len(v) for v in sets["rcasr_bm25_w2"].values()], dtype=float)
    ci_r, ci_f = arm["vs_bm25_w3"]["gv_recall_delta_ci"], arm["vs_bm25_w3"]["fa_per_image_delta_ci"]
    gates = {"W1_recall_noninferior": {"delta": arm["vs_bm25_w3"]["gv_recall_delta"], "ci": ci_r,
                                        "margin": -MARGIN, "passed": bool(ci_r[0] > -MARGIN)},
             "W2_fewer_false_alarms": {"delta": arm["vs_bm25_w3"]["fa_per_image_delta"], "ci": ci_f,
                                        "passed": bool(ci_f[1] < 0)}}
    claim = gates["W1_recall_noninferior"]["passed"] and gates["W2_fewer_false_alarms"]["passed"]
    gates["review_workload_claim"] = {"passed": bool(claim), "rule": "both W1 and W2 must pass"}
    evidence = ctx["records"]
    per_image = read_json(CS_ROOT / V2 / "rcasr_results.json")["evidence"]["per_image_mean"]
    result = {
        "plan": PLAN, "snapshot_sha256": sha256(OUT / "input_snapshot.json"),
        "target_width": TARGET_WIDTH, "realised": {
            "width_mean": float(widths.mean()), "width_median": float(np.median(widths)),
            "width_p95": float(np.quantile(widths, .95)), "empty_selection_rate": float((widths == 0).mean()),
            "judged_pairs_total": int(widths.sum()), "width_3_pairs_total": sum(map(len, sets["rcasr_bm25_w3"].values()))},
        "judge": {"source": f"{JUDGE}/internal_pairs.jsonl", "new_pairs_judged": len(missing),
                  "n_verdicts_used": len(needed),
                  "parse_errors_among_used": sum(1 for p in needed if verdicts[p] is None),
                  "tokens_per_pair": "not recorded per pair in the judge cache; judge cost scales with judged pairs per image"},
        "extractor_tokens_per_image": {k: v for k, v in per_image.items() if "tokens" in k},
        "gates": gates, "arms": arms,
        "diagnostic_curve_no_gate": {w: {k: m[k] for k in ("width_mean", "gv_recall", "false_alarms_per_image",
                                                             "misses_per_image", "need_review_rate")}
                                     for w, m in ((w, rx.e2e_metrics(s, verdicts, statuses, family))
                                                  for w, s in diagnostic.items())},
        "limits": ["all images development-exposed; retrospective cross-fitting, not an independent test",
                   "judge verdicts are the width-3 E2 cache (two serving stacks, 11 parse errors) reused unchanged",
                   "closed-world precision treats unrecorded pairs as non-violations; assumption-dependent",
                   "the diagnostic curve is post-hoc and supports no claim; the arm is width 2 only"],
        "n_evidence_images": len(evidence)}
    write_json(OUT / "lowwidth_results.json", result)
    with (OUT / "lowwidth_rows.jsonl").open("w", encoding="utf-8") as handle:
        for i in sorted(sets["rcasr_bm25_w2"]):
            handle.write(json.dumps({"image_id": i, **{n: s[i] for n, s in sets.items()}}) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        print(f"froze -> {OUT / 'input_snapshot.json'}")
    if args.run:
        result = run()
        print(json.dumps(result["gates"], indent=2))
        print(json.dumps({k: {m: v[m] for m in ("width_mean", "gv_recall", "false_alarms_per_image")}
                          for k, v in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
