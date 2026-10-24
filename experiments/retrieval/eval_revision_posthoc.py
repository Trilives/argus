#!/usr/bin/env python3
"""Revision post-hoc analyses H1, H2, H4, H5 and H6 (Buildings 4625629, major revision).

Pre-specification: ``docs/paper_v3/design/REVISION_POSTHOC_PRESPEC.md`` (written before this
script ran). Everything is computed from frozen outputs: no model is called and no reported
number is replaced. H3, H7 and H8 are in ``eval_revision_cases.py``, which shares this freeze.

Usage::

    uv run --group analysis python experiments/retrieval/eval_revision_posthoc.py --prepare
    uv run --group analysis python experiments/retrieval/eval_revision_posthoc.py --run
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

CS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CS_ROOT / "src"))
sys.path.insert(0, str(CS_ROOT / "experiments" / "retrieval"))

import eval_lowwidth_e2 as lw  # noqa: E402
import rcasr_experiment as rx  # noqa: E402
from rcasr import features, recall, score_rows, select_at, width_threshold  # noqa: E402
from research_snapshot import read_json, sha256, snapshot, verify_snapshot  # noqa: E402
from set_selection import matched_rank_sets  # noqa: E402
from set_selection_metrics import cluster_index, paired_intervals  # noqa: E402

OUT = CS_ROOT / "results/2026-10-24_revision_posthoc"
PLAN = "docs/paper_v3/design/REVISION_POSTHOC_PRESPEC.md"
V2 = "results/2026-09-22_rcasr_v2"
LW = "results/2026-09-22_rcasr_lowwidth"
JUDGE = "results/2026-09-22_rcasr_judge"
PUBLIC = "results/2026-09-22_rcasr_public"
ROBOT_RUNS = {
    "frozen_threshold": ("deploy/Output/dog-20260923/reports/dog0910-28.json",
                         "deploy/Output/dog-20260923/reports/dog0910-29.json"),
    "site_recalibrated": ("deploy/Output/dog-20260923/reports/dog0910r-28.json",
                          "deploy/Output/dog-20260923/reports/dog0910r-29.json"),
    "fp8_on_robot": ("deploy/Output/dog-20260927/reports/dog0910f-28.json",
                     "deploy/Output/dog-20260927/reports/dog0910f-29.json")}
FILES = (PLAN, "experiments/retrieval/eval_revision_posthoc.py",
         "experiments/retrieval/eval_revision_cases.py", "experiments/retrieval/eval_lowwidth_e2.py",
         "experiments/retrieval/eval_rcasr_public.py",
         f"{V2}/input_snapshot.json", f"{V2}/rcasr_rows.jsonl", f"{V2}/rcasr_results.json",
         f"{LW}/input_snapshot.json", f"{LW}/lowwidth_rows.jsonl", f"{LW}/lowwidth_results.json",
         f"{JUDGE}/input_snapshot.json", f"{JUDGE}/internal_pairs.jsonl", f"{JUDGE}/public_pairs.jsonl",
         f"{PUBLIC}/input_snapshot.json", f"{PUBLIC}/final_config.json", f"{PUBLIC}/public_selection.json",
         f"{PUBLIC}/atom_evidence.jsonl", f"{PUBLIC}/facts.jsonl",
         "data/public_eval/public_labels_test.jsonl", "data/public_eval/public_map_v1.json",
         "data/splits/site_keys_candidate.json", "data/rules/proposed/subject_screen_v2.json",
         *(p for paths in ROBOT_RUNS.values() for p in paths), *rx.ANALYSIS_FILES)
NON_VISUAL = "not_available_for_single_image"
E2E_KEYS = ("gv_recall", "gv_precision_closed_world", "false_alarms_per_image", "misses_per_image",
            "need_review_rate", "width_mean")
DECISIONS = ("violation", "need_review", "clear", "clear_empty")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_jsonl(path: Path) -> dict:
    return {r["image_id"]: r for r in map(json.loads, path.read_text().splitlines())}


def prepare() -> None:
    if (OUT / "input_snapshot.json").exists():
        raise FileExistsError("existing freeze cannot be overwritten")
    frozen = snapshot(CS_ROOT, [CS_ROOT / f for f in FILES])
    frozen.update(plan_path=PLAN, created_at=datetime.now(timezone.utc).isoformat(),
                  scope="post-hoc revision analyses over frozen outputs; no model calls; "
                        "retrospective, not an independent test")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "input_snapshot.json", frozen)


def load() -> dict:
    """Verified frozen state shared by both revision scripts."""
    verify_snapshot(CS_ROOT, read_json(OUT / "input_snapshot.json"))
    v2_frozen = read_json(CS_ROOT / V2 / "input_snapshot.json")
    verify_snapshot(CS_ROOT, v2_frozen)
    ctx = rx.context(CS_ROOT, v2_frozen)
    ctx["_grids"] = rx.evidence_grids(ctx)
    rules = read_json(CS_ROOT / "data/rules/rules_en.json")
    verdicts = lw.load_verdicts(CS_ROOT / JUDGE / "internal_pairs.jsonl")
    if (CS_ROOT / LW / "extra_pairs.jsonl").exists():
        verdicts.update(lw.load_verdicts(CS_ROOT / LW / "extra_pairs.jsonl"))
    v2_rows = read_jsonl(CS_ROOT / V2 / "rcasr_rows.jsonl")
    lw_rows = read_jsonl(CS_ROOT / LW / "lowwidth_rows.jsonl")
    sets = {name: {i: row[name] for i, row in lw_rows.items()}
            for name in ("rcasr_bm25_w2", "rcasr_bm25_w3", "bm25_w3", "bm25_w2_matched")}
    sets["r4_top3"] = {i: row["r4_top3"] for i, row in v2_rows.items()}
    sets["rcasr_r4"] = {i: row["selected"]["rcasr_r4"] for i, row in v2_rows.items()}
    return {"ctx": ctx, "rules": rules, "verdicts": verdicts, "v2_rows": v2_rows, "sets": sets,
            "v2": read_json(CS_ROOT / V2 / "rcasr_results.json"),
            "statuses": rx.gold_statuses(CS_ROOT),
            "family": {r["rule_id"]: r["rule_id"].split("-")[1] for r in rules},
            "units": rx.site_units(ctx), "cfg": ctx["config"]["gates"]}


def bootstrap(rows: np.ndarray, stat, seed: int, repeats: int, level: float = .95) -> list[float]:
    """Percentile interval of ``stat`` over unit-resampled column sums."""
    rng = np.random.default_rng(seed)
    draws = rows[rng.integers(0, len(rows), (repeats, len(rows)))].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = stat(draws)
    values = values[np.isfinite(values)]
    tail = (1 - level) / 2
    return [float(x) for x in np.quantile(values, [tail, 1 - tail])]


def diff_ratio(draws: np.ndarray) -> np.ndarray:
    return (draws[:, 0] - draws[:, 1]) / draws[:, 2]


# --- H1 non-visual provisions -------------------------------------------------------------

def non_visual(rules: list[dict]) -> set[str]:
    return {r["rule_id"] for r in rules if str(r.get("visual_screening_rule", "")).strip() == NON_VISUAL}


def drop(sets: dict, banned: set[str]) -> dict:
    return {i: [r for r in rs if r not in banned] for i, rs in sets.items()}


def e2e(state: dict, sets: dict) -> dict:
    m = rx.e2e_metrics(sets, state["verdicts"], state["statuses"], state["family"])
    return {k: m[k] for k in E2E_KEYS}


def masked(scores: dict, banned: set[str]) -> dict:
    return {i: {r: (-math.inf if r in banned else s) for r, s in row.items()} for i, row in scores.items()}


def rethreshold(state: dict, width: float, banned: set[str]) -> dict:
    """Fold-fitted RCASR-BM25 scores; per-fold threshold re-set on the fitting folds at ``width``."""
    ctx = state["ctx"]
    variant = rx.variants("bm25", ctx["config"])[0]
    chosen = {}
    for fold in state["v2"]["arms"]["rcasr_bm25"]["folds"]:
        fitting, _, test = rx.fold_roles(ctx["split"]["folds"], ctx["split"]["k"], fold["held_out"])
        params = {"theta": tuple(fold["theta"]), "tau": fold["tau"], "tau_open": fold["tau_open"]}
        fit_scores = masked(rx.scored(ctx, variant, ctx["_grids"], params, fitting), banned)
        test_scores = masked(rx.scored(ctx, variant, ctx["_grids"], params, test), banned)
        chosen.update(select_at(test_scores, width_threshold(fit_scores, width)))
    return chosen


def decide(verdicts: list[dict]) -> str:
    labels = [v.get("label") for v in verdicts]
    if not labels:
        return "clear_empty"
    if "non_compliant" in labels:
        return "violation"
    if any(label in ("need_review", None) for label in labels):
        return "need_review"
    return "clear"


def robot_summary(frames: list[dict], banned: set[str]) -> dict:
    decisions, judged, review = Counter(), 0, 0
    for frame in frames:
        kept = [v for v in frame.get("verdicts") or [] if v["rule_id"] not in banned]
        decisions[decide(kept)] += 1
        judged += len(kept)
        review += sum(v.get("label") in ("need_review", None) for v in kept)
    n = len(frames)
    return {"frames": n, "decision_share": {d: decisions[d] / n for d in DECISIONS},
            "judged_pairs_per_frame": judged / n,
            "review_share_of_judged_pairs": review / judged if judged else None}


def robot(banned: set[str]) -> dict:
    out = {}
    for name, paths in ROBOT_RUNS.items():
        frames = [f for p in paths for f in read_json(CS_ROOT / p)["frames"]]
        mismatched = [f["image_id"] for f in frames if decide(f.get("verdicts") or []) != f["decision"]]
        if mismatched:
            raise AssertionError(f"{name}: decision rule does not reproduce {len(mismatched)} frames")
        selecting = Counter(r for f in frames for r in f.get("selected") or [] if r in banned)
        out[name] = {"before": robot_summary(frames, set()), "after": robot_summary(frames, banned),
                     "frames_selecting_non_visual": dict(selecting)}
    return out


def h1(state: dict) -> dict:
    banned = non_visual(state["rules"])
    sets = state["sets"]
    removed_after = {arm: {"before": e2e(state, sets[arm]), "after": e2e(state, drop(sets[arm], banned)),
                           "pairs_removed": sum(r in banned for rs in sets[arm].values() for r in rs)}
                     for arm in ("rcasr_bm25_w2", "rcasr_bm25_w3", "bm25_w3")}
    for width, arm in ((2.0, "rcasr_bm25_w2"), (3.0, "rcasr_bm25_w3")):
        if rethreshold(state, width, set()) != sets[arm]:
            raise AssertionError(f"rethreshold does not reproduce {arm}")
    positives = {i: state["ctx"]["positives"][i] for i in sets["bm25_w3"]}
    removed_before = {}
    for width in (2.0, 3.0):
        chosen = rethreshold(state, width, banned)
        unjudged = sorted((i, r) for i, rs in chosen.items() for r in rs if (i, r) not in state["verdicts"])
        removed_before[str(width)] = {
            "retrieval_recall": recall(chosen, positives),
            "retrieval_recall_original": recall(sets[f"rcasr_bm25_w{int(width)}"], positives),
            "width_mean": sum(map(len, chosen.values())) / len(chosen),
            "unjudged_pairs": len(unjudged),
            "end_to_end": None if unjudged else e2e(state, chosen)}
    return {"non_visual_provisions": sorted(banned), "removed_after_selection": removed_after,
            "removed_before_threshold": removed_before, "robot_round": robot(banned)}


# --- H2 recorded applicable but not violated ----------------------------------------------

def coverage_rows(units: list, left: dict, right: dict, statuses: dict, status: str) -> np.ndarray:
    rows = []
    for ids in units:
        row = [0.0, 0.0, 0.0]
        for i in ids:
            pairs = {r for r, s in statuses[i].items() if status == "any" or s == status}
            row[0] += len(pairs & set(left[i]))
            row[1] += len(pairs & set(right[i]))
            row[2] += len(pairs)
        rows.append(row)
    return np.array(rows, dtype=float)


def h2(state: dict) -> dict:
    sets, units, cfg = state["sets"], state["units"], state["cfg"]
    statuses = {i: state["statuses"][i] for i in sets["bm25_w3"]}
    out = {"support": {}, "coverage": {}, "paired": {}}
    for status in ("compliant", "undetermined", "non_compliant", "any"):
        out["support"][status] = int(coverage_rows(units, sets["bm25_w3"], sets["bm25_w3"], statuses, status)[:, 2].sum())
        out["coverage"][status] = {}
        for arm, chosen in sets.items():
            rows = coverage_rows(units, chosen, chosen, statuses, status)
            out["coverage"][status][arm] = float(rows[:, 0].sum() / rows[:, 2].sum())
        for left in ("rcasr_bm25_w3", "rcasr_bm25_w2"):
            rows = coverage_rows(units, sets[left], sets["bm25_w3"], statuses, status)
            out["paired"].setdefault(status, {})[f"{left}_minus_bm25_w3"] = {
                "delta": float((rows[:, 0].sum() - rows[:, 1].sum()) / rows[:, 2].sum()),
                "ci95": bootstrap(rows, diff_ratio, cfg["bootstrap_seed"], cfg["bootstrap_repeats"])}
    return out


# --- H4 further CRC operating points ------------------------------------------------------

def h4(state: dict) -> dict:
    ctx = state["ctx"]
    out = {}
    check = rx.calibration_arm(ctx, rx.Variant("bm25", "bm25"), ctx["_grids"], .10, False)
    frozen = state["v2"]["calibration"]["bm25_pooled_0.1"]
    if abs(check["width_mean"] - frozen["width_mean"]) > 1e-9:
        raise AssertionError("calibration_arm does not reproduce the frozen BM25 pooled alpha 0.10 point")
    for cap, alphas in ((3, (.30, .40)), (5, (.10, .20, .30))):
        sel = {**ctx["config"]["selection"], "cap": cap}
        cctx = {**ctx, "config": {**ctx["config"], "selection": sel}}
        for prior in ("bm25", "r4"):
            for alpha in alphas:
                for grouped in (False, True):
                    key = f"{prior}_{'group' if grouped else 'pooled'}_a{alpha:.2f}_cap{cap}"
                    out[key] = rx.calibration_arm(cctx, rx.Variant(prior, prior), ctx["_grids"], alpha, grouped)
    return out


# --- H5 statistics ------------------------------------------------------------------------

def recall_rows(left: dict, right: dict, positives: dict, units: list) -> np.ndarray:
    """Same grouping and row order as ``paired_intervals``, so the draws coincide."""
    groups: dict[str, list[str]] = {}
    for image_id, group in cluster_index(sorted(left), units).items():
        groups.setdefault(group, []).append(image_id)
    return np.array([[sum(len(set(left[i]) & positives[i]) - len(set(right[i]) & positives[i]) for i in ids),
                      sum(len(positives[i]) for i in ids)] for _, ids in sorted(groups.items())], dtype=float)


def ratio(draws: np.ndarray) -> np.ndarray:
    return np.where(draws[:, 1] > 0, draws[:, 0] / np.where(draws[:, 1] > 0, draws[:, 1], 1), np.nan)


def gate_pairs(state: dict) -> dict:
    rows = state["v2_rows"]
    pick = lambda f: {i: f(r) for i, r in rows.items()}
    return {"P1": (pick(lambda r: r["selected"]["rcasr_bm25"]), pick(lambda r: r["reference"]["bm25"])),
            "P2": (pick(lambda r: r["selected"]["rcasr_r4"]), pick(lambda r: r["reference"]["r4"])),
            "S1": (pick(lambda r: r["selected"]["rcasr_bm25"]), pick(lambda r: r["selected"]["adaptive_bm25"])),
            "S2": (pick(lambda r: r["selected"]["rcasr_bm25"]), pick(lambda r: r["selected"]["ternary_bm25"]))}


def h5a(state: dict) -> dict:
    cfg, units = state["cfg"], state["units"]
    reported = {"P1": "P1_rcasr_bm25_vs_bm25", "P2": "P2_rcasr_r4_vs_r4",
                "S1": "S1_structure_beyond_adaptivity", "S2": "S2_continuous_beyond_ternary"}
    out = {}
    for gate, (left, right) in gate_pairs(state).items():
        positives = {i: state["ctx"]["positives"][i] for i in left}
        rows = recall_rows(left, right, positives, units)
        ci95 = bootstrap(rows, ratio, cfg["bootstrap_seed"], cfg["bootstrap_repeats"])
        frozen = state["v2"]["gates"][reported[gate]]["recall_delta_ci"]
        if max(abs(a - b) for a, b in zip(ci95, frozen)) > 1e-12:
            raise AssertionError(f"{gate}: re-bootstrap does not reproduce the reported interval")
        out[gate] = {"ci95": ci95, "ci_bonferroni_98_75": bootstrap(
            rows, ratio, cfg["bootstrap_seed"], cfg["bootstrap_repeats"], level=1 - .05 / 4)}
    out["all_four_exclude_zero_at_98_75"] = all(v["ci_bonferroni_98_75"][0] > 0 for v in out.values())
    return out


def mc_error(intervals: list[list[float]]) -> dict:
    array = np.array(intervals)
    return {"lower_sd": float(array[:, 0].std(ddof=1)), "upper_sd": float(array[:, 1].std(ddof=1)),
            "lower_range": [float(array[:, 0].min()), float(array[:, 0].max())],
            "upper_range": [float(array[:, 1].min()), float(array[:, 1].max())]}


def h5b(state: dict) -> dict:
    cfg, units = state["cfg"], state["units"]
    seed, repeats = cfg["bootstrap_seed"], cfg["bootstrap_repeats"]
    left, right = gate_pairs(state)["P1"]
    positives = {i: state["ctx"]["positives"][i] for i in left}
    rows = recall_rows(left, right, positives, units)
    w2, w3 = state["sets"]["rcasr_bm25_w2"], state["sets"]["bm25_w3"]
    w1 = lambda s, n: lw.paired_e2e_ci(units, w2, w3, state["verdicts"], state["statuses"], s, n)["gv_recall_delta_ci"]
    frozen = read_json(CS_ROOT / LW / "lowwidth_results.json")["gates"]["W1_recall_noninferior"]["ci"]
    if max(abs(a - b) for a, b in zip(w1(seed, repeats), frozen)) > 1e-12:
        raise AssertionError("W1: re-bootstrap does not reproduce the reported interval")
    seeds = [seed + k for k in range(1, 21)]
    return {"seeds": seeds, "repeats": repeats,
            "P1": {**mc_error([bootstrap(rows, ratio, s, repeats) for s in seeds]),
                   "ci95_20000": bootstrap(rows, ratio, seed, 20000)},
            "W1_gv_recall": {**mc_error([w1(s, repeats) for s in seeds]), "ci95_20000": w1(seed, 20000)}}


def sign_test(wins: int, losses: int) -> float:
    n, k = wins + losses, min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def h5c(state: dict) -> dict:
    out = {}
    for arm, ref in (("rcasr_bm25", "bm25_matched"), ("rcasr_r4", "r4_matched")):
        tally = Counter()
        for row in state["v2"]["per_site"]:
            a, b = row[arm], row[ref]
            tally["win" if a is not None and a > b + 1e-12 else
                  "loss" if a is not None and a < b - 1e-12 else "tie"] += 1
        out[arm] = {**tally, "sign_test_p_two_sided": sign_test(tally["win"], tally["loss"])}
    return out


def family_rows(units: list, left: dict, right: dict, positives: dict) -> np.ndarray:
    return np.array([[sum(len(set(left[i]) & positives[i]) for i in ids),
                      sum(len(set(right[i]) & positives[i]) for i in ids),
                      sum(len(positives[i]) for i in ids)] for ids in units], dtype=float)


def h5d(state: dict) -> dict:
    ctx, cfg, units = state["ctx"], state["cfg"], state["units"]
    out = {}
    for fam in sorted(set(state["family"].values())):
        run = rx.crossfit(ctx, rx.Variant(f"lofo_{fam}_bm25", "bm25", ignore_family=fam), ctx["_grids"])
        reference = rx.matched_reference(ctx, rx.Variant("ref", "bm25"), run["selected"])
        rules = {r for r, f in state["family"].items() if f == fam}
        positives = {i: ctx["positives"][i] & rules for i in run["selected"]}
        rows = family_rows(units, run["selected"], reference, positives)
        frozen = state["v2"]["family_holdout"]["bm25"][fam]
        if abs(recall(run["selected"], positives) - frozen["held_out_family_recall"]) > 1e-12:
            raise AssertionError(f"LOFO {fam} does not reproduce the reported recall")
        out[fam] = {"held_out_family_recall": recall(run["selected"], positives),
                    "matched_reference_recall": recall(reference, positives),
                    "support": int(rows[:, 2].sum()),
                    "delta_ci95": bootstrap(rows, diff_ratio, cfg["bootstrap_seed"], cfg["bootstrap_repeats"])}
    return out


def e2e_family_rows(units: list, left: dict, right: dict, state: dict, fam: str) -> np.ndarray:
    rows = []
    for ids in units:
        row = [0.0, 0.0, 0.0]
        for i in ids:
            truth = {r for r, s in state["statuses"][i].items()
                     if s == "non_compliant" and state["family"][r] == fam}
            for col, chosen in ((0, left), (1, right)):
                row[col] += sum(r in truth and state["verdicts"][(i, r)] == "non_compliant" for r in chosen[i])
            row[2] += len(truth)
        rows.append(row)
    return np.array(rows, dtype=float)


def h5e(state: dict) -> dict:
    cfg, units, sets = state["cfg"], state["units"], state["sets"]
    out = {}
    for left in ("rcasr_bm25_w2", "rcasr_bm25_w3"):
        for fam in sorted(set(state["family"].values())):
            rows = e2e_family_rows(units, sets[left], sets["bm25_w3"], state, fam)
            out.setdefault(left, {})[fam] = {
                "recall": float(rows[:, 0].sum() / rows[:, 2].sum()),
                "bm25_w3_recall": float(rows[:, 1].sum() / rows[:, 2].sum()),
                "support": int(rows[:, 2].sum()),
                "delta_ci95": bootstrap(rows, diff_ratio, cfg["bootstrap_seed"], cfg["bootstrap_repeats"])}
    return out


def h5f(state: dict) -> dict:
    ids = sorted(state["sets"]["bm25_w3"])
    recorded = sum(len(state["ctx"]["positives"][i]) for i in ids)
    base = read_json(CS_ROOT / LW / "lowwidth_results.json")["arms"]["bm25_w3"]["gv_recall"]
    return {"recorded_violations": recorded, "margin": lw.MARGIN,
            "margin_in_violations": lw.MARGIN * recorded,
            "margin_share_of_bm25_w3_gv_recall": lw.MARGIN / base, "bm25_w3_gv_recall": base}


# --- H6 the 154 images without a site -----------------------------------------------------

def h6(state: dict) -> dict:
    ctx, cfg = state["ctx"], state["cfg"]
    final = read_json(CS_ROOT / PUBLIC / "final_config.json")
    site_ids = sorted(ctx["split"]["folds"])
    other_ids = sorted(set(ctx["priors"]["bm25"]) - set(site_ids))
    feats = {i: features(ctx["compiled"], ctx["posts"][i], final["tau"], final["tau_open"], ctx["opening"])
             for i in ctx["priors"]["bm25"]}
    scores = score_rows(ctx["priors"]["bm25"], feats, tuple(final["theta"]))
    site_scores, other_scores = rx.subset(scores, site_ids), rx.subset(scores, other_ids)
    site_pos = {i: ctx["positives"][i] for i in site_ids}
    other_pos = {i: ctx["positives"][i] for i in other_ids}
    resub = recall(select_at(site_scores, width_threshold(site_scores, 3.0)), site_pos)
    if abs(resub - final["internal_resubstitution_recall"]) > 1e-12:
        raise AssertionError("final configuration does not reproduce its resubstitution recall")
    out = {"n_images": len(other_ids), "recorded_violations": sum(map(len, other_pos.values())),
           "images_with_violation": sum(bool(p) for p in other_pos.values()), "widths": {}}
    for width in (3.0, 2.0):
        chosen = select_at(other_scores, width_threshold(site_scores, width))
        reference = matched_rank_sets(rx.subset(ctx["priors"]["bm25"], other_ids), sum(map(len, chosen.values())))
        ci = paired_intervals(chosen, reference, other_pos, ctx["clusters"],
                              seed=cfg["bootstrap_seed"], repeats=cfg["bootstrap_repeats"])
        out["widths"][str(width)] = {
            "rcasr_recall": recall(chosen, other_pos), "bm25_matched_recall": recall(reference, other_pos),
            "width_mean": sum(map(len, chosen.values())) / len(chosen),
            "empty_rate": sum(not v for v in chosen.values()) / len(chosen),
            "recall_delta_ci95": ci["recall_delta_ci"], "bootstrap_unit": ci["unit"]}
    count = lambda pos: dict(Counter(state["family"][r] for p in pos.values() for r in p))
    out["violations_by_family"] = {"site_346": count(site_pos), "unsited_154": count(other_pos)}
    index = cluster_index(sorted(ctx["priors"]["bm25"]), ctx["clusters"])
    site_roots = {index[i] for i in site_ids}
    out["unsited_sharing_component_with_site_image"] = sum(index[i] in site_roots for i in other_ids)
    return out


def run() -> dict:
    state = load()
    result = {"plan": PLAN, "snapshot_sha256": sha256(OUT / "input_snapshot.json"),
              "H1_non_visual": h1(state), "H2_recorded_not_violated": h2(state), "H4_crc": h4(state),
              "H5_statistics": {"a_multiplicity": h5a(state), "b_monte_carlo": h5b(state),
                                "c_per_site_sign_test": h5c(state), "d_lofo_ci": h5d(state),
                                "e_family_e2e": h5e(state), "f_margin": h5f(state)},
              "H6_unsited_154": h6(state),
              "limits": ["post hoc: all primary results were seen before this plan",
                         "all internal images development-exposed; retrospective, not an independent test",
                         "H6 is not site-independent; H2 says nothing about unrecorded applicable pairs"]}
    write_json(OUT / "posthoc_results.json", result)
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
        print(json.dumps(result["H1_non_visual"]["robot_round"], indent=2))


if __name__ == "__main__":
    main()
