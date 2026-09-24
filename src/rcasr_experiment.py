"""Hash-bound analysis stages for RCASR: cross-fitting, arms, gates, ablations, judge.

The analysis freeze pins the evidence freeze's output plus every file that scores it.
``evaluate`` makes no model call; ``judge`` makes model calls but reads no gold.
Pre-specification: ``docs/paper_v3/design/RCASR_PRESPEC.md``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

from atom_posterior import read_evidence
from rcasr import (features, fit, logit, r4_prior, recall, score_rows, select_at,
                   width_threshold)
from research_snapshot import read_json, sha256, snapshot, verify_snapshot
from set_selection import calibrate, capped_selection, matched_rank_sets, select
from set_selection_metrics import metrics, paired_intervals
from set_selection_report import load_inputs
from site_partition import partition
from typed_predicates import atoms
from typed_schema import compile_schema

R4_PATH = "results/retrieval/agent_grep_visual_eval.json"
SITES_PATH = "data/splits/site_keys_confirmed.json"
ANALYSIS_FILES = (
    "src/atom_posterior.py", "src/rcasr.py", "src/rcasr_experiment.py",
    "src/site_partition.py", "src/set_selection.py", "src/set_selection_metrics.py",
    "src/set_selection_report.py", "src/typed_schema.py", "src/typed_predicates.py",
    "src/research_snapshot.py", "src/retrieval/bm25.py", "src/retrieval/base.py",
    "experiments/retrieval/eval_rcasr.py", "tests/test_rcasr.py", "tests/test_atom_posterior.py",
    "tests/test_site_partition.py", SITES_PATH, R4_PATH, "data/rules/rules_en.json",
    "data/rules/proposed/typed_evidence_v1.json", "data/annotations/image_rule_gold.json",
    "data/rule_assets/rule_index.json", "results/retrieval/gold_facts_generic_en.json",
    "results/data_audit/near_duplicates.json", "pyproject.toml", "uv.lock")


@dataclass(frozen=True)
class Variant:
    """One cross-fitted arm: a prior, a grid restriction and a selection rule."""
    name: str
    prior: str
    grid: dict = field(default_factory=dict)
    evidence: str = "soft"
    fixed_k: bool = False
    ignore_family: str | None = None


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


# --- freeze -------------------------------------------------------------------------------

def prepare(root: Path, out: Path, evidence_dir: str, config_path: str) -> None:
    if (out / "input_snapshot.json").exists():
        raise FileExistsError("existing freeze cannot be overwritten; use a new output directory")
    config = read_json(root / config_path)
    ev = root / evidence_dir
    names = [config_path, config["plan_path"], *config["plan_addenda"], *ANALYSIS_FILES]
    files = [root / n for n in names] + [ev / "input_snapshot.json", ev / "atom_evidence.jsonl"]
    frozen = snapshot(root, files)
    frozen.update(config_path=config_path, evidence_dir=evidence_dir,
                  scope="346 site-assigned development-exposed images, site-grouped cross-fitting; "
                        "retrospective, not an independent test",
                  created_at=datetime.now(timezone.utc).isoformat())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "input_snapshot.json", frozen)


# --- inputs -------------------------------------------------------------------------------

def compiled_overlay(root: Path, config: dict) -> dict:
    return compile_schema(read_json(root / "data/rules/rules_en.json"),
                          read_json(root / config["source_overlay"]))


def merged_posteriors(records: dict) -> dict:
    """``image -> atom -> posterior | None`` over both passes; atoms never asked are absent."""
    out = {}
    for image_id, row in records.items():
        posts = {}
        for name in ("subject", "state"):
            posts.update(row[name]["posteriors"])
        out[image_id] = posts
    return out


def shuffled(posts: dict, seed: int) -> dict:
    ids = sorted(posts)
    donors = list(ids)
    random.Random(seed).shuffle(donors)
    return {image_id: posts[donor] for image_id, donor in zip(ids, donors)}


def tau_pairs(grid: dict) -> list[tuple[float, float]]:
    if "tau_pairs" in grid:
        return [tuple(p) for p in grid["tau_pairs"]]
    return [(t, o) for t in grid["tau"] for o in grid["tau_open"]]


def feature_grid(compiled: dict, posts: dict, pairs: list, opening: set[str],
                 ternary: bool = False) -> dict:
    return {pair: {i: features(compiled, p, pair[0], pair[1], opening, ternary)
                   for i, p in posts.items()} for pair in pairs}


def r4_ranked(root: Path) -> dict[str, list[str]]:
    rows = read_json(root / R4_PATH)["rows"]
    return {row["image_id"]: [r["rule_id"] for r in sorted(row["retrieved"], key=lambda r: r["rank"])]
            for row in rows}


def context(root: Path, frozen: dict) -> dict:
    """Everything evaluate needs, loaded once and checked for population agreement."""
    config = read_json(root / frozen["config_path"])
    ev = root / frozen["evidence_dir"]
    ev_frozen = read_json(ev / "input_snapshot.json")
    records = read_evidence(ev / "atom_evidence.jsonl", ev_frozen, sha256(ev / "input_snapshot.json"))
    bm25, positives, family, clusters = load_inputs(root)
    library = sorted(family)
    ranked = r4_ranked(root)
    if set(records) != set(bm25) or set(ranked) != set(bm25):
        raise ValueError("evidence, BM25 and R4 populations differ")
    split = partition(read_json(root / SITES_PATH), clusters, set(bm25),
                      config["folds"]["k"], config["folds"]["seed"])
    return {"config": config, "compiled": compiled_overlay(root, config), "records": records,
            "posts": merged_posteriors(records), "positives": positives, "family": family,
            "clusters": clusters, "split": split, "library": library,
            "priors": {"bm25": bm25, "r4": {i: r4_prior(ranked[i], library) for i in bm25}},
            "r4_top3": {i: ranked[i][:3] for i in bm25},
            "opening": set(config["abstention"]["opening_atoms"])}


# --- cross-fitting ------------------------------------------------------------------------

def fold_roles(folds: dict[str, int], k: int, held_out: int) -> tuple[list, list, list]:
    calibration = (held_out + 1) % k
    fitting = sorted(i for i, f in folds.items() if f not in (held_out, calibration))
    return (fitting, sorted(i for i, f in folds.items() if f == calibration),
            sorted(i for i, f in folds.items() if f == held_out))


def budget(ctx: dict, prior: str, ids: list[str]) -> float:
    """w*: mean width of the fixed k = 3 reference on the fitting population."""
    if prior == "bm25":
        return 3.0
    return sum(len(ctx["r4_top3"][i]) for i in ids) / len(ids)


def subset(rows: dict, ids: list[str]) -> dict:
    return {i: rows[i] for i in ids}


def fit_positives(ctx: dict, ids: list[str], ignore_family: str | None) -> dict:
    drop = {r for r, f in ctx["family"].items() if f == ignore_family}
    return {i: ctx["positives"][i] - drop for i in ids}


def fold_fit(ctx: dict, variant: Variant, grids: dict, fitting: list[str]) -> dict:
    grid = {**ctx["config"]["grid"], **variant.grid}
    pairs = tau_pairs(grid)
    table = {pair: subset(grids[variant.evidence][pair], fitting) for pair in pairs}
    width = budget(ctx, variant.prior, fitting)
    params = fit(subset(ctx["priors"][variant.prior], fitting), table,
                 fit_positives(ctx, fitting, variant.ignore_family), width,
                 {**grid, "tau_pairs": pairs})
    scores = score_rows(subset(ctx["priors"][variant.prior], fitting),
                        table[(params["tau"], params["tau_open"])], params["theta"])
    return {**params, "width": width, "threshold": width_threshold(scores, width)}


def scored(ctx: dict, variant: Variant, grids: dict, params: dict, ids: list[str]) -> dict:
    pair = (params["tau"], params["tau_open"])
    return score_rows(subset(ctx["priors"][variant.prior], ids),
                      subset(grids[variant.evidence][pair], ids), params["theta"])


def crossfit(ctx: dict, variant: Variant, grids: dict) -> dict:
    """Held-out selections, scores and per-fold parameters for one arm."""
    folds, k = ctx["split"]["folds"], ctx["split"]["k"]
    selected, scores, per_fold = {}, {}, []
    for held_out in range(k):
        fitting, calibration, test = fold_roles(folds, k, held_out)
        params = fold_fit(ctx, variant, grids, fitting)
        test_scores = scored(ctx, variant, grids, params, test)
        if variant.fixed_k:
            size = math.ceil(params["width"] - 1e-9)
            chosen = {i: select(row, 0.)[:size] for i, row in test_scores.items()}
        else:
            chosen = select_at(test_scores, params["threshold"])
        selected.update(chosen)
        scores.update(test_scores)
        per_fold.append({"held_out": held_out, "calibration_fold": (held_out + 1) % k,
                         "n_fitting": len(fitting), "n_test": len(test),
                         **{key: params[key] for key in ("theta", "tau", "tau_open", "width",
                                                         "threshold", "recall")},
                         "calibration_ids": calibration})
    return {"selected": selected, "scores": scores, "folds": per_fold}


# --- comparison ---------------------------------------------------------------------------

def site_units(ctx: dict) -> list[list[str]]:
    return [u["images"] for u in ctx["split"]["units"]]


def compare(ctx: dict, left: dict, right: dict, cfg: dict) -> dict:
    ids = sorted(left)
    positives = {i: ctx["positives"][i] for i in ids}
    ci = paired_intervals(left, right, positives, site_units(ctx),
                          seed=cfg["bootstrap_seed"], repeats=cfg["bootstrap_repeats"])
    ci["unit"] = "confirmed site unit (sites fused through near-duplicate clusters)"
    ci["interpretation"] = "site-clustered bootstrap over cross-fitted held-out predictions; retrospective"
    return {"recall_delta": recall(left, positives) - recall(right, positives), "uncertainty": ci}


def matched_reference(ctx: dict, variant: Variant, selected: dict) -> dict:
    total = sum(map(len, selected.values()))
    return matched_rank_sets(subset(ctx["priors"][variant.prior], sorted(selected)), total)


def arm_summary(ctx: dict, variant: Variant, run: dict) -> dict:
    ids = sorted(run["selected"])
    positives = {i: ctx["positives"][i] for i in ids}
    reference = matched_reference(ctx, variant, run["selected"])
    return {"variant": {"prior": variant.prior, "grid": variant.grid, "evidence": variant.evidence,
                        "fixed_k": variant.fixed_k, "ignore_family": variant.ignore_family},
            "metrics": metrics(run["selected"], positives, ctx["family"]),
            "reference_metrics": metrics(reference, positives, ctx["family"]),
            "vs_matched_prior": compare(ctx, run["selected"], reference, ctx["config"]["gates"]),
            "folds": [{k: v for k, v in f.items() if k != "calibration_ids"} for f in run["folds"]]}


def gate(summary: dict, bar: float | None) -> dict:
    delta, ci = summary["recall_delta"], summary["uncertainty"]["recall_delta_ci"]
    passed = ci is not None and ci[0] > 0 and (bar is None or delta >= bar)
    return {"recall_delta": delta, "recall_delta_ci": ci, "bar": bar, "passed": bool(passed)}


# --- variants -----------------------------------------------------------------------------

def variants(prior: str, config: dict) -> list[Variant]:
    zero = {"theta_gate": [0], "theta_state": [0], "tau_pairs": [(0, 0)]}
    same_tau = {"tau_pairs": [(t, t) for t in config["grid"]["tau"]]}
    base = Variant(f"rcasr_{prior}", prior)
    return [base,
            Variant(f"adaptive_{prior}", prior, zero),
            Variant(f"ternary_{prior}", prior, evidence="hard"),
            Variant(f"no_gate_{prior}", prior, {"theta_gate": [0]}),
            Variant(f"no_state_{prior}", prior, {"theta_state": [0]}),
            Variant(f"no_abstention_{prior}", prior, {"tau_pairs": [(0, 0)]}),
            Variant(f"no_open_class_{prior}", prior, same_tau),
            replace(base, name=f"fixed_k_{prior}", fixed_k=True),
            Variant(f"null_{prior}", prior, evidence="null")]


def evidence_grids(ctx: dict) -> dict:
    pairs = tau_pairs(ctx["config"]["grid"])
    posts, compiled, opening = ctx["posts"], ctx["compiled"], ctx["opening"]
    null = shuffled(posts, ctx["config"]["gates"]["null_seed"])
    return {"soft": feature_grid(compiled, posts, pairs, opening),
            "hard": feature_grid(compiled, posts, pairs, opening, ternary=True),
            "null": feature_grid(compiled, null, pairs, opening)}


# --- calibration (CRC on the calibration fold only) ---------------------------------------

def top_family(ctx: dict, row: dict) -> str:
    """Label-free group: the family of the image's top-scored rule."""
    return ctx["family"][select(row, 0.)[0]]


def group_thresholds(ctx: dict, cal_scores: dict, alpha: float, min_size: int) -> dict:
    positives = {i: ctx["positives"][i] for i in cal_scores}
    pooled = calibrate(cal_scores, positives, alpha)["threshold"]
    groups: dict[str, list[str]] = {}
    for image_id, row in cal_scores.items():
        groups.setdefault(top_family(ctx, row), []).append(image_id)
    out = {"__pooled__": pooled}
    for name, ids in groups.items():
        if len(ids) >= min_size:
            out[name] = calibrate(subset(cal_scores, ids), subset(positives, ids), alpha)["threshold"]
    return out


def calibration_arm(ctx: dict, variant: Variant, grids: dict, alpha: float, grouped: bool) -> dict:
    folds, k = ctx["split"]["folds"], ctx["split"]["k"]
    cap, min_size = ctx["config"]["selection"]["cap"], ctx["config"]["selection"]["group_min_size"]
    rows, thresholds = {}, []
    for held_out in range(k):
        fitting, calibration, test = fold_roles(folds, k, held_out)
        params = fold_fit(ctx, variant, grids, fitting)
        table = group_thresholds(ctx, scored(ctx, variant, grids, params, calibration), alpha,
                                 min_size if grouped else 10 ** 9)
        thresholds.append(table)
        for image_id, row in scored(ctx, variant, grids, params, test).items():
            threshold = table.get(top_family(ctx, row), table["__pooled__"]) if grouped else table["__pooled__"]
            rows[image_id] = capped_selection(row, threshold, cap)
    return summarise_calibration(ctx, rows, alpha, thresholds)


def summarise_calibration(ctx: dict, rows: dict, alpha: float, thresholds: list) -> dict:
    ids = sorted(rows)
    selected = {i: rows[i]["selected"] for i in ids}
    m = metrics(selected, {i: ctx["positives"][i] for i in ids}, ctx["family"])
    deferred = sum(r["status"] == "need_review" for r in rows.values())
    return {"alpha": alpha, "held_out_empirical_risk": m["risk"], "risk_gap": m["risk"] - alpha,
            "width_mean": m["width_mean"], "width_median": m["width_median"],
            "width_p95": m["width_p95"], "recall_micro": m["recall_micro"],
            "deferral_rate": deferred / len(ids),
            "infeasible_folds": sum(t["__pooled__"] is None for t in thresholds),
            "r2_triggered": bool(alpha == .10 and m["width_mean"] > 3),
            "note": "risk and width are pre-cap; deferral = whole image to need_review"}


# --- E5 and harm --------------------------------------------------------------------------

def per_site(ctx: dict, arms: dict[str, dict]) -> list[dict]:
    rows = []
    for unit in ctx["split"]["units"]:
        ids = unit["images"]
        support = sum(len(ctx["positives"][i]) for i in ids)
        row = {"unit": unit["unit"], "n_images": len(ids), "recorded_violations": support}
        for name, selected in arms.items():
            hits = sum(len(set(selected[i]) & ctx["positives"][i]) for i in ids)
            row[name] = hits / support if support else None
        rows.append(row)
    return sorted(rows, key=lambda r: (-r["n_images"], r["unit"]))


def family_holdout(ctx: dict, prior: str, grids: dict) -> dict:
    out = {}
    for fam in sorted(set(ctx["family"].values())):
        run = crossfit(ctx, Variant(f"lofo_{fam}_{prior}", prior, ignore_family=fam), grids)
        reference = matched_reference(ctx, Variant("ref", prior), run["selected"])
        rules = {r for r, f in ctx["family"].items() if f == fam}
        positives = {i: ctx["positives"][i] & rules for i in run["selected"]}
        out[fam] = {"held_out_family_recall": recall(run["selected"], positives),
                    "matched_prior_family_recall": recall(reference, positives),
                    "positive_support": sum(map(len, positives.values())),
                    "fitted_params": [{k: f[k] for k in ("theta", "tau", "tau_open")}
                                      for f in run["folds"]]}
    return out


def harm(ctx: dict, arm: dict, reference: dict) -> dict:
    opening = ctx["opening"]
    lost = [(i, r) for i in sorted(arm) for r in sorted(set(reference[i]) - set(arm[i]))
            if r in ctx["positives"][i]]
    gained = [(i, r) for i in sorted(arm) for r in sorted(set(arm[i]) - set(reference[i]))
              if r in ctx["positives"][i]]
    touches = lambda r: bool(atoms(ctx["compiled"]["rules"][r]["gate"]) & opening)
    return {"lost": len(lost), "gained": len(gained),
            "lost_by_family": dict(Counter(ctx["family"][r] for _, r in lost)),
            "gained_by_family": dict(Counter(ctx["family"][r] for _, r in gained)),
            "lost_with_opening_class_gate": sum(touches(r) for _, r in lost),
            "lost_pairs": [{"image_id": i, "rule_id": r} for i, r in lost]}


def evidence_audit(ctx: dict) -> dict:
    """Unknown rates, argmax/text agreement and cost, over all 500 evidence rows."""
    unknown, agree, total, cost = Counter(), 0, 0, Counter()
    for row in ctx["records"].values():
        for name in ("subject", "state"):
            item = row[name]
            cost[f"{name}_seconds"] += item["seconds"]
            cost[f"{name}_prompt_tokens"] += item["prompt_tokens"] or 0
            cost[f"{name}_completion_tokens"] += item["completion_tokens"] or 0
            cost[f"{name}_parse_errors"] += item["status"] != "ok"
            for atom, post in item["posteriors"].items():
                if post is None:
                    unknown[atom] += 1
                    continue
                total += 1
                agree += max(("yes", "no", "unclear"), key=lambda w: post[f"p_{w}"]) == item["answers"].get(atom)
    n = len(ctx["records"])
    return {"n_images": n, "unknown_posteriors": dict(unknown.most_common()),
            "argmax_text_agreement": agree / total if total else None,
            "per_image_mean": {k: v / n for k, v in cost.items()},
            "model_calls_per_image": 2}


# --- orchestration ------------------------------------------------------------------------

def primary_gates(ctx: dict, runs: dict) -> dict:
    bar = ctx["config"]["gates"]["recall_delta"]
    cfg = ctx["config"]["gates"]
    s = lambda name: runs[name]["selected"]
    return {"P1_rcasr_bm25_vs_bm25": gate(runs["rcasr_bm25"]["summary"]["vs_matched_prior"], bar),
            "P2_rcasr_r4_vs_r4": gate(runs["rcasr_r4"]["summary"]["vs_matched_prior"], bar),
            "S1_structure_beyond_adaptivity": gate(compare(ctx, s("rcasr_bm25"), s("adaptive_bm25"), cfg), None),
            "S2_continuous_beyond_ternary": gate(compare(ctx, s("rcasr_bm25"), s("ternary_bm25"), cfg), None),
            "N_null_bm25": gate(runs["null_bm25"]["summary"]["vs_matched_prior"], bar)}


def evaluate(root: Path, out: Path) -> tuple[dict, dict]:
    frozen = read_json(out / "input_snapshot.json")
    verify_snapshot(root, frozen)
    ctx = context(root, frozen)
    grids = evidence_grids(ctx)
    runs = {}
    for prior in ("bm25", "r4"):
        for variant in variants(prior, ctx["config"]):
            run = crossfit(ctx, variant, grids)
            run["summary"] = arm_summary(ctx, variant, run)
            runs[variant.name] = run
    held = sorted(runs["rcasr_bm25"]["selected"])
    references = {p: matched_reference(ctx, Variant("ref", p), runs[f"rcasr_{p}"]["selected"])
                  for p in ("bm25", "r4")}
    calibration = {f"{p}_{'group' if g else 'pooled'}_{a}": calibration_arm(ctx, Variant(p, p), grids, a, g)
                   for p in ("bm25", "r4") for a in ctx["config"]["selection"]["alphas"]
                   for g in (False, True)}
    results = {
        "scope": frozen["scope"], "snapshot_sha256": sha256(out / "input_snapshot.json"),
        "population": ctx["split"]["counts"], "gates": primary_gates(ctx, runs),
        "arms": {name: run["summary"] for name, run in runs.items()},
        "calibration": calibration,
        "per_site": per_site(ctx, {"rcasr_bm25": runs["rcasr_bm25"]["selected"],
                                   "bm25_matched": references["bm25"],
                                   "rcasr_r4": runs["rcasr_r4"]["selected"],
                                   "r4_matched": references["r4"]}),
        "family_holdout": {p: family_holdout(ctx, p, grids) for p in ("bm25", "r4")},
        "harm": {p: harm(ctx, runs[f"rcasr_{p}"]["selected"], references[p]) for p in ("bm25", "r4")},
        "evidence": evidence_audit(ctx),
        "limits": ["all images development-exposed; retrospective cross-fitting, not an independent test",
                   "opening-class atoms were chosen from an all-500 harm table (not fold-blind)",
                   "recorded violations are the only positives; width is an exposure proxy, not exposure"],
    }
    rows = {i: {"fold": ctx["split"]["folds"][i],
                "selected": {name: run["selected"][i] for name, run in runs.items()},
                "reference": {p: references[p][i] for p in references},
                "r4_top3": ctx["r4_top3"][i]} for i in held}
    return results, rows


# --- E2: one frozen per-pair judge over the union of selected pairs -----------------------

E2_ARMS = ("rcasr_bm25", "adaptive_bm25", "rcasr_r4")


def e2_sets(rows: dict) -> dict[str, dict[str, list[str]]]:
    sets = {name: {i: r["selected"][name] for i, r in rows.items()} for name in E2_ARMS}
    sets["bm25_matched"] = {i: r["reference"]["bm25"] for i, r in rows.items()}
    sets["r4_matched"] = {i: r["reference"]["r4"] for i, r in rows.items()}
    sets["r4_top3"] = {i: r["r4_top3"] for i, r in rows.items()}
    return sets


def judge_pair(backend, image, facts: list[str], rule_id: str) -> dict:
    import symbolic_judgement  # noqa: PLC0415
    from pipeline import extract_rule_evidence  # noqa: PLC0415

    evidence = extract_rule_evidence(backend, image, facts, [rule_id])
    verdict = symbolic_judgement.symbolic_verdict(evidence[0]) if evidence else {}
    label = (verdict.get("compliance_judgement") or {}).get("compliance_label")
    return {"label": label, "parse_error": evidence[0].get("parse_error") if evidence else "no_evidence",
            "evidence": evidence[0] if evidence else None}


def judge(root: Path, out: Path, model: str, workers: int = 8) -> None:
    """J3-sym per pair on the frozen judge model; append-only cache, resumable, no gold read."""
    from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415
    from threading import Lock  # noqa: PLC0415

    from backends.openai_api import OpenAIBackend  # noqa: PLC0415
    from images import load_image  # noqa: PLC0415

    frozen = read_json(out / "input_snapshot.json")
    verify_snapshot(root, frozen)
    rows = {json.loads(l)["image_id"]: json.loads(l) for l in (out / "rcasr_rows.jsonl").read_text().splitlines()}
    facts = read_json(root / "results/retrieval/gold_facts_generic_en.json")
    pairs = sorted({(i, r) for s in e2_sets(rows).values() for i, rs in s.items() for r in rs})
    cache = out / "judge_pairs.jsonl"
    done = {(d["image_id"], d["rule_id"]) for d in map(json.loads, cache.read_text().splitlines())} if cache.exists() else set()
    backend, lock = OpenAIBackend(model), Lock()
    names = {Path(n).stem: n for n in read_json(root / "data/annotations/image_rule_gold.json")["images"]}

    def work(pair):
        image = load_image(root / "data/images" / names[pair[0]])
        return pair, judge_pair(backend, image, facts[pair[0]]["image_facts"], pair[1])

    todo = [p for p in pairs if p not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, future in enumerate(as_completed([pool.submit(work, p) for p in todo]), 1):
            (image_id, rule_id), item = future.result()
            with lock, cache.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"image_id": image_id, "rule_id": rule_id, "model": model,
                                         **item}, ensure_ascii=False) + "\n")
            if n % 50 == 0 or n == len(todo):
                print(f"judged {len(done) + n}/{len(pairs)}", flush=True)


def gold_statuses(root: Path) -> dict:
    return {i: row["rule_statuses"] for i, row in
            read_json(root / "data/annotations/image_rule_gold.json")["gold"].items()}


def e2e_metrics(selected: dict, verdicts: dict, statuses: dict, family: dict) -> dict:
    ids = sorted(selected)
    predicted = {i: {r for r in selected[i] if verdicts[(i, r)] == "non_compliant"} for i in ids}
    truth = {i: {r for r, s in statuses[i].items() if s == "non_compliant"} for i in ids}
    tp = sum(len(predicted[i] & truth[i]) for i in ids)
    n_pred, n_true = sum(map(len, predicted.values())), sum(map(len, truth.values()))
    fa = Counter(statuses[i].get(r, "unrecorded") for i in ids for r in predicted[i] - truth[i])
    review = sum(verdicts[(i, r)] in ("need_review", None) for i in ids for r in selected[i])
    fam_recall = {}
    for fam in sorted(set(family.values())):
        t = sum(len({r for r in truth[i] if family[r] == fam}) for i in ids)
        h = sum(len({r for r in predicted[i] & truth[i] if family[r] == fam}) for i in ids)
        fam_recall[fam] = h / t if t else None
    p = tp / n_pred if n_pred else None
    r = tp / n_true if n_true else None
    return {"gv_recall": r, "gv_precision_closed_world": p,
            "gv_f1_closed_world": 2 * p * r / (p + r) if p and r else None,
            "false_alarms_per_image": sum(fa.values()) / len(ids), "false_alarm_gold_status": dict(fa),
            "misses_per_image": (n_true - tp) / len(ids),
            "need_review_rate": review / max(1, sum(map(len, selected.values()))),
            "exact_image_success": sum(predicted[i] == truth[i] for i in ids) / len(ids),
            "macro_family_recall": fam_recall, "width_mean": sum(map(len, selected.values())) / len(ids),
            "precision_note": "closed-world precision treats unrecorded pairs as non-violations; "
                              "assumption-dependent, not an estimate of true precision"}


def evaluate_e2e(root: Path, out: Path) -> dict:
    rows = {json.loads(l)["image_id"]: json.loads(l) for l in (out / "rcasr_rows.jsonl").read_text().splitlines()}
    judged = [json.loads(l) for l in (out / "judge_pairs.jsonl").read_text().splitlines()]
    verdicts = {(d["image_id"], d["rule_id"]): d["label"] for d in judged}
    statuses, family = gold_statuses(root), {r["rule_id"]: r["rule_id"].split("-")[1]
                                             for r in read_json(root / "data/rules/rules_en.json")}
    sets = e2_sets(rows)
    return {"judge_model": sorted({d["model"] for d in judged}),
            "n_pairs_judged": len(verdicts),
            "parse_errors": sum(bool(d["parse_error"]) for d in judged),
            "arms": {name: e2e_metrics(s, verdicts, statuses, family) for name, s in sets.items()},
            "scope": "346 site images, cross-fitted held-out selections; one frozen per-pair judge"}
