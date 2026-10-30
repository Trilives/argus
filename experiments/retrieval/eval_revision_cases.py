#!/usr/bin/env python3
"""Revision post-hoc analyses H3, H7 and H8 (Buildings 4625629, major revision).

Pre-specification: ``docs/research/design/REVISION_POSTHOC_PRESPEC.md``. Shares the freeze written
by ``eval_revision_posthoc.py --prepare``; no model is called.

Usage::

    uv run --group analysis python experiments/retrieval/eval_revision_cases.py --run
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import statistics

import eval_revision_posthoc as rp
import eval_rcasr_public as pub
import rcasr_experiment as rx
from atom_posterior import read_evidence
from public_map import image_targets
from rcasr import (features, literal_table, logit, probability, score_rows, select_at, tie_break,
                   width_threshold, LOG_FLOOR)
from research_snapshot import read_json, sha256
from set_selection import matched_rank_sets
from typed_predicates import atoms, comparisons

CS_ROOT = rp.CS_ROOT
OUTRANKED = -0.1  # evidence terms above this: the prior alone left the pair out
CONFIDENT = 0.8
F6_IMAGES = ("img_0116", "img_0369", "img_0452", "img_1479")
F6_MISS_RULE = "R-OPN-001-horizontal-opening-protection"
PUBLIC_RULE = "rule_4"
PROJECT_TYPES = (("residential", ("花园", "住宅", "公寓", "家园", "苑")),
                 ("office_or_commercial", ("大厦", "广场", "商业", "写字楼")),
                 ("industrial_or_technology_park", ("科技园", "产业园", "工业园", "园区")),
                 ("cultural_or_public_venue", ("馆", "学校", "医院", "中心")))


def confidence(post: dict | None) -> dict | None:
    if post is None or post["p_yes"] + post["p_no"] <= 0:
        return None
    q = post["p_yes"] / (post["p_yes"] + post["p_no"])
    return {"q": q, "u": post["p_unclear"], "c": (1 - post["p_unclear"]) * max(q, 1 - q)}


def literals(spec: dict, name: str, table: dict) -> list[float]:
    """P(expected value) of each occurrence of a known atom in the gate and screening formulas."""
    leaves = [leaf for key in ("gate", "visual") if spec[key] for leaf in comparisons(spec[key])]
    return [table[name] if expected else 1 - table[name]
            for _, atom, op, expected in leaves if atom == name and op == "==" and name in table]


def role(spec: dict, name: str) -> str:
    where = [key for key in ("gate", "visual") if spec[key] and name in atoms(spec[key])]
    return {("gate",): "gate", ("visual",): "screening"}.get(tuple(where), "both")


def log_term(tree: tuple | None, table: dict) -> float:
    return max(math.log(LOG_FLOOR), math.log(max(probability(tree, table), LOG_FLOOR)))


# --- H3 decomposition ---------------------------------------------------------------------

def decompose(spec: dict, posts: dict, prior: float, image_id: str, rule: str, params: dict,
              opening: set[str], threshold: float) -> dict:
    """Score terms, the pivotal atom and its effect for one image-provision pair."""
    theta1, theta2 = params["theta"]
    table = literal_table(posts, params["tau"], params["tau_open"], opening)

    def score(tab: dict) -> tuple[float, float, float]:
        gate, vis = theta1 * log_term(spec["gate"], tab), theta2 * log_term(spec["visual"], tab)
        z = logit(prior) + gate + vis + tie_break(image_id, rule)
        return gate, vis, 1 / (1 + math.exp(-z)) if z > -700 else 0.0

    gate, vis, s = score(table)
    names = sorted((atoms(spec["gate"]) if spec["gate"] else set()) |
                   (atoms(spec["visual"]) if spec["visual"] else set()))
    rows = []
    for name in names:
        lifted = score({k: v for k, v in table.items() if k != name})[2] if name in table else s
        rows.append({"atom": name, "role": role(spec, name), "opening_class": name in opening,
                     "known": name in table, "literals": literals(spec, name, table),
                     **(confidence(posts.get(name)) or {"q": None, "u": None, "c": None}),
                     "score_if_unknown": lifted, "lift": lifted - s})
    known_rows = [r for r in rows if r["known"]]
    pivot = max(known_rows, key=lambda r: (r["lift"], r["atom"])) if known_rows else None
    if gate + vis > OUTRANKED:
        category = "outranked"
    elif pivot is not None and pivot["c"] >= CONFIDENT:
        category = "confident_contradiction"
    else:
        category = "uncertain_atom_read_as_evidence"
    return {"image_id": image_id, "rule_id": rule, "prior_logit": logit(prior), "gate_term": gate,
            "state_term": vis, "score": s, "threshold": threshold, "score_minus_threshold": s - threshold,
            "category": category, "pivotal_atom": pivot["atom"] if pivot else None,
            "pivot_selects_alone": bool(pivot and pivot["score_if_unknown"] >= threshold),
            "pivot": pivot, "atoms": rows}


def with_rank(row: dict, scores: dict) -> dict:
    ordered = sorted(scores, key=lambda r: (-scores[r], r))
    return {**row, "rank_in_image": ordered.index(row["rule_id"]) + 1}


def fold_params(state: dict, arm: str) -> dict[int, dict]:
    return {f["held_out"]: f for f in state["v2"]["arms"][arm]["folds"]}


def h3a(state: dict) -> dict:
    ctx = state["ctx"]
    ids = sorted(ctx["split"]["folds"])
    out = {}
    for arm in ("rcasr_bm25", "rcasr_r4"):
        out[arm] = []
        for held_out, fold in sorted(fold_params(state, arm).items()):
            tally = Counter()
            for i in ids:
                for name, post in ctx["posts"][i].items():
                    cls = "opening" if name in ctx["opening"] else "other"
                    tau = fold["tau_open"] if cls == "opening" else fold["tau"]
                    conf = confidence(post)
                    tally[f"{cls}_total"] += 1
                    if conf is not None and conf["c"] >= tau:
                        tally[f"{cls}_known"] += 1
                        tally[f"{cls}_known_below_{CONFIDENT}"] += conf["c"] < CONFIDENT
            out[arm].append({"held_out": held_out, "theta": fold["theta"], "tau": fold["tau"],
                             "tau_open": fold["tau_open"],
                             **{f"{c}_known_share": tally[f"{c}_known"] / tally[f"{c}_total"]
                                for c in ("opening", "other")},
                             **{f"{c}_known_with_c_below_{CONFIDENT}": tally[f"{c}_known_below_{CONFIDENT}"] /
                                max(1, tally[f"{c}_known"]) for c in ("opening", "other")}})
    return out


def internal_pair(state: dict, image_id: str, rule: str, threshold_width: float | None) -> dict:
    ctx = state["ctx"]
    held_out = ctx["split"]["folds"][image_id]
    fold = fold_params(state, "rcasr_bm25")[held_out]
    params = {"theta": tuple(fold["theta"]), "tau": fold["tau"], "tau_open": fold["tau_open"]}
    threshold = fold["threshold"]
    variant = rx.variants("bm25", ctx["config"])[0]
    if threshold_width is not None:
        fitting, _, _ = rx.fold_roles(ctx["split"]["folds"], ctx["split"]["k"], held_out)
        threshold = width_threshold(rx.scored(ctx, variant, ctx["_grids"], params, fitting), threshold_width)
    scores = rx.scored(ctx, variant, ctx["_grids"], params, [image_id])[image_id]
    row = decompose(ctx["compiled"]["rules"][rule], ctx["posts"][image_id], ctx["priors"]["bm25"][image_id][rule],
                    image_id, rule, params, ctx["opening"], threshold)
    if abs(row["score"] - scores[rule]) > 1e-12:
        raise AssertionError(f"decomposition does not reproduce the score of {image_id}/{rule}")
    return {**with_rank(row, scores), "held_out_fold": held_out}


def tally(rows: list[dict], state_family: dict, opening_gate) -> dict:
    return {"by_category": dict(Counter(r["category"] for r in rows)),
            "by_category_and_family": dict(Counter(f"{r['category']}|{state_family[r['rule_id']]}" for r in rows)),
            "by_category_opening_gate": dict(Counter(f"{r['category']}|{opening_gate(r['rule_id'])}" for r in rows)),
            "pivot_opening_class": dict(Counter(bool(r["pivot"] and r["pivot"]["opening_class"]) for r in rows)),
            "pivot_selects_alone": sum(r["pivot_selects_alone"] for r in rows),
            # added after the prespec: direction of the pivot (literal >= .5 supports the violation)
            "by_category_pivot_supports_violation": dict(Counter(
                f"{r['category']}|{bool(r['pivot'] and min(r['pivot']['literals'], default=0) >= .5)}" for r in rows))}


def h3b(state: dict) -> dict:
    ctx = state["ctx"]
    lost = state["v2"]["harm"]["bm25"]["lost_pairs"]
    rows = [internal_pair(state, p["image_id"], p["rule_id"], None) for p in lost]
    opening_gate = lambda r: bool(atoms(ctx["compiled"]["rules"][r]["gate"]) & ctx["opening"])
    return {"n_lost": len(rows), **tally(rows, state["family"], opening_gate), "pairs": rows}


def public_state(state: dict) -> dict:
    """Public scores under the frozen configuration; must reproduce the frozen selection."""
    out = CS_ROOT / rp.PUBLIC
    frozen = read_json(out / "input_snapshot.json")
    records = read_evidence(out / "atom_evidence.jsonl", frozen, sha256(out / "input_snapshot.json"))
    posts = rx.merged_posteriors(records)
    final = read_json(out / "final_config.json")
    bm25, _ = pub.public_scores(out)
    ctx = state["ctx"]
    feats = {i: features(ctx["compiled"], posts[i], final["tau"], final["tau_open"], ctx["opening"]) for i in bm25}
    scores = score_rows(bm25, feats, tuple(final["theta"]))
    threshold = width_threshold(scores, pub.WIDTH)
    rcasr = select_at(scores, threshold)
    reference = matched_rank_sets(bm25, sum(map(len, rcasr.values())))
    stored = read_json(out / "public_selection.json")
    if any(stored[i]["rcasr_bm25"] != rcasr[i] or stored[i]["bm25_matched"] != reference[i] for i in stored):
        raise AssertionError("public scores do not reproduce the frozen public selection")
    return {"posts": posts, "bm25": bm25, "scores": scores, "threshold": threshold, "rcasr": rcasr,
            "reference": reference, "params": {"theta": tuple(final["theta"]), "tau": final["tau"],
                                               "tau_open": final["tau_open"]}}


def public_pair(state: dict, ps: dict, image_id: str, rule: str) -> dict:
    ctx = state["ctx"]
    row = decompose(ctx["compiled"]["rules"][rule], ps["posts"][image_id], ps["bm25"][image_id][rule],
                    image_id, rule, ps["params"], ctx["opening"], ps["threshold"])
    if abs(row["score"] - ps["scores"][image_id][rule]) > 1e-12:
        raise AssertionError(f"decomposition does not reproduce the public score of {image_id}/{rule}")
    return {**with_rank(row, ps["scores"][image_id]), "selected": rule in ps["rcasr"][image_id],
            "in_bm25_matched": rule in ps["reference"][image_id]}


def h3c(state: dict, ps: dict) -> dict:
    mapping, gold = read_json(pub.MAP), pub.labels()
    items = [t for i in sorted(ps["rcasr"]) for t in image_targets(mapping, gold[i])
             if t["status"] == "scoreable" and t["public_rule"] == PUBLIC_RULE]
    rows = [public_pair(state, ps, t["image_id"], r) for t in items for r in t["targets"]]
    missed = [r for r in rows if not r["selected"]]
    ctx = state["ctx"]
    opening_gate = lambda r: bool(atoms(ctx["compiled"]["rules"][r]["gate"]) & ctx["opening"])
    return {"n_positives": len(items), "n_pairs": len(rows), "selected": len(rows) - len(missed),
            "missed": {"n": len(missed), **tally(missed, state["family"], opening_gate)},
            "pivotal_atoms_of_missed": dict(Counter(r["pivotal_atom"] for r in missed)), "pairs": rows}


def h3d(state: dict, public_rows: list[dict]) -> dict:
    sets, statuses = state["sets"], state["statuses"]
    f6 = [i for i in F6_IMAGES if statuses[i].get(F6_MISS_RULE) == "non_compliant"
          and F6_MISS_RULE in sets["bm25_w3"][i] and F6_MISS_RULE not in sets["rcasr_bm25_w2"][i]]
    if len(f6) != 1:
        raise AssertionError(f"F6 panel (b) rule matches {len(f6)} images, expected 1")
    case = next((r for r in sorted(public_rows, key=lambda r: (r["image_id"], r["rule_id"]))
                 if r["in_bm25_matched"] and not r["selected"]), None)
    return {"f6_panel_b": internal_pair(state, f6[0], F6_MISS_RULE, 2.0), "public_rule_4_first": case}


# --- H7 data description ------------------------------------------------------------------

def project_type(name: str) -> str:
    return next((kind for kind, words in PROJECT_TYPES if any(w in name for w in words)), "unspecified")


def spread(values: list[int]) -> dict:
    return {"min": min(values), "median": statistics.median(values), "max": max(values),
            "mean": sum(values) / len(values)}


def status_counts(statuses: dict, ids: list[str], key) -> dict:
    out: dict[str, Counter] = {}
    for i in ids:
        for rule, status in statuses[i].items():
            out.setdefault(key(rule), Counter())[status] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def h7(state: dict) -> dict:
    statuses = state["statuses"]
    all_ids = sorted(statuses)
    site_ids = sorted(state["ctx"]["split"]["folds"])
    sites = read_json(CS_ROOT / rx.SITES_PATH)["sites"]
    images_per_site = [sum(im["image_id"] in set(site_ids) for im in s["images"]) for s in sites]
    kinds = Counter()
    kind_images = Counter()
    for s, n in zip(sites, images_per_site):
        kinds[project_type(s["site_key"])] += 1
        kind_images[project_type(s["site_key"])] += n
    out = {"project_type_rule": "first keyword match on the site key; names are not output",
           "sites": len(sites), "project_type_sites": dict(kinds), "project_type_images": dict(kind_images),
           "images_per_site": spread(images_per_site), "populations": {}}
    for label, ids in (("site_346", site_ids), ("all_500", all_ids)):
        violations = [sum(s == "non_compliant" for s in statuses[i].values()) for i in ids]
        out["populations"][label] = {
            "n_images": len(ids), "recorded_pairs": sum(len(statuses[i]) for i in ids),
            "violations_per_image": spread(violations),
            "violations_per_image_histogram": dict(sorted(Counter(violations).items())),
            "images_without_recorded_violation": sum(v == 0 for v in violations),
            "by_family": status_counts(statuses, ids, lambda r: state["family"][r]),
            "by_provision": status_counts(statuses, ids, lambda r: r)}
    return out


# --- H8 atom vocabulary -------------------------------------------------------------------

def h8(state: dict) -> dict:
    ctx = state["ctx"]
    config = ctx["config"]
    subject = set(read_json(CS_ROOT / config["passes"]["subject"]["questions_from"])["questions"])
    statepass = set(config["passes"]["state"]["questions"])
    provisions = {}
    used: dict[str, list[str]] = {}
    for rule, spec in sorted(ctx["compiled"]["rules"].items()):
        gate = sorted(atoms(spec["gate"])) if spec["gate"] else []
        screen = sorted(atoms(spec["visual"])) if spec["visual"] else []
        provisions[rule] = {"gate_atoms": gate, "screening_atoms": screen}
        for name in set(gate) | set(screen):
            used.setdefault(name, []).append(rule)
    vocabulary = sorted(subject | statepass | set(used))
    return {"subject_pass_atoms": len(subject), "state_pass_atoms": len(statepass),
            "asked_atoms": len(subject | statepass), "both_passes": sorted(subject & statepass),
            "formula_atoms_never_asked": sorted(set(used) - subject - statepass),
            "asked_atoms_unused": sorted((subject | statepass) - set(used)),
            "opening_class": sorted(ctx["opening"]), "provisions": provisions,
            "atoms": {a: {"pass": "subject" if a in subject else "state" if a in statepass else "not_asked",
                          "opening_class": a in ctx["opening"], "provisions": sorted(used.get(a, []))}
                      for a in vocabulary}}


def run() -> dict:
    state = rp.load()
    ps = public_state(state)
    c = h3c(state, ps)
    result = {"plan": rp.PLAN, "snapshot_sha256": sha256(rp.OUT / "input_snapshot.json"),
              "H3_atom_errors": {"a_known_shares": h3a(state), "b_lost_pairs": h3b(state),
                                 "c_public_rule_4": c, "d_cases": h3d(state, c["pairs"])},
              "H7_data": h7(state), "H8_atoms": h8(state),
              "limits": ["post hoc: all primary results were seen before this plan",
                         "pivotal atom is a one-at-a-time attribution, not a causal estimate",
                         "project type is a keyword rule over site names, not a survey"]}
    rp.write_json(rp.OUT / "cases_results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--run", action="store_true")
    if parser.parse_args().run:
        result = run()
        print(json.dumps(result["H3_atom_errors"]["b_lost_pairs"]["by_category"], indent=2))


if __name__ == "__main__":
    main()
