#!/usr/bin/env python3
"""§3.4 no-subject default: frozen ``compliant`` (A) vs abstain ``need_review`` (B).

Pre-specification: ``docs/paper_v3/design/S34_DEFAULT_PRESPEC.md`` (written before this
script ran). No model call: both arms recompute J3-sym over the Stage-4 evidence stored in
the frozen judge cache, differing only in the defaults table that the gate-unknown branch
reads. Check 1 (A reproduces the cache) and check 2 (identical ``non_compliant`` pairs) are
exact; the review-load numbers are descriptive.

Usage::

    uv run --group analysis python experiments/retrieval/eval_s34_default.py --prepare
    uv run --group analysis python experiments/retrieval/eval_s34_default.py --run
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

CS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CS_ROOT / "src"))

import rcasr_experiment as rx  # noqa: E402
from research_snapshot import read_json, sha256, snapshot, verify_snapshot  # noqa: E402
import symbolic_judgement  # noqa: E402

OUT = CS_ROOT / "results/2026-09-23_s34_default"
PLAN = "docs/paper_v3/design/S34_DEFAULT_PRESPEC.md"
JUDGE = "results/2026-09-22_rcasr_judge"
V2 = "results/2026-09-22_rcasr_v2"
LOWWIDTH = "results/2026-09-22_rcasr_lowwidth"
FILES = ("experiments/retrieval/eval_s34_default.py", PLAN, "src/symbolic_judgement.py",
         "src/rcasr_experiment.py", "data/rules/rules_en.json", "data/annotations/image_rule_gold.json",
         f"{JUDGE}/input_snapshot.json", f"{JUDGE}/internal_pairs.jsonl", f"{JUDGE}/public_pairs.jsonl",
         f"{V2}/rcasr_rows.jsonl", f"{LOWWIDTH}/lowwidth_rows.jsonl")
ARMS = {"A_frozen_compliant": "compliant", "B_abstain_need_review": "need_review"}


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare() -> None:
    if (OUT / "input_snapshot.json").exists():
        raise FileExistsError("existing freeze cannot be overwritten")
    frozen = snapshot(CS_ROOT, [CS_ROOT / f for f in FILES])
    frozen.update(plan_path=PLAN, created_at=datetime.now(timezone.utc).isoformat(),
                  scope="frozen E2 judge cache (internal + public pairs); symbolic recomputation only")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "input_snapshot.json", frozen)
    (OUT / "analysis_plan.md").write_bytes((CS_ROOT / PLAN).read_bytes())


def relabel(rows: list[dict], default: str) -> tuple[dict, Counter]:
    """``(image_id, rule_id) -> label`` under ``default``, plus the route histogram."""
    rule_ids = [r["rule_id"] for r in read_json(CS_ROOT / "data/rules/rules_en.json")]
    symbolic_judgement._DEFAULTS = dict.fromkeys(rule_ids, default)
    labels, routes = {}, Counter()
    for row in rows:
        key = (row["image_id"], row["rule_id"])
        if row["evidence"] is None:
            labels[key], _ = row["label"], routes.update(["no_evidence"])
            continue
        verdict = symbolic_judgement.symbolic_verdict(row["evidence"])
        labels[key] = verdict["compliance_judgement"]["compliance_label"]
        routes[verdict["symbolic"]["route"]] += 1
    symbolic_judgement._DEFAULTS = None
    return labels, routes


def compare(rows: list[dict]) -> dict:
    stored = {(r["image_id"], r["rule_id"]): r["label"] for r in rows}
    arms = {name: relabel(rows, default) for name, default in ARMS.items()}
    a, b = arms["A_frozen_compliant"][0], arms["B_abstain_need_review"][0]
    mismatch = sorted(k for k in stored if a[k] != stored[k])
    flagged = {name: {k for k, v in labels.items() if v == "non_compliant"} for name, (labels, _) in arms.items()}
    rerouted = sorted(k for k in a if a[k] != b[k])
    return {"n_pairs": len(rows), "check1_a_reproduces_cache": not mismatch, "check1_mismatches": mismatch[:20],
            "check2_identical_non_compliant": flagged["A_frozen_compliant"] == flagged["B_abstain_need_review"],
            "n_non_compliant": len(flagged["A_frozen_compliant"]),
            "routes": dict(arms["A_frozen_compliant"][1]),
            "rerouted": {"n": len(rerouted), "share": len(rerouted) / len(rows),
                         "transitions": dict(Counter(f"{a[k]}->{b[k]}" for k in rerouted))},
            "labels": {"A": a, "B": b}, "rerouted_keys": rerouted}


def run() -> dict:
    frozen = read_json(OUT / "input_snapshot.json")
    verify_snapshot(CS_ROOT, frozen)
    load = lambda name: [json.loads(l) for l in (CS_ROOT / JUDGE / name).read_text().splitlines()]  # noqa: E731
    internal, public = compare(load("internal_pairs.jsonl")), compare(load("public_pairs.jsonl"))
    passed = all(c["check1_a_reproduces_cache"] and c["check2_identical_non_compliant"] for c in (internal, public))

    statuses = rx.gold_statuses(CS_ROOT)
    family = {r["rule_id"]: r["rule_id"].split("-")[1] for r in read_json(CS_ROOT / "data/rules/rules_en.json")}
    v2_rows = {json.loads(l)["image_id"]: json.loads(l) for l in (CS_ROOT / V2 / "rcasr_rows.jsonl").read_text().splitlines()}
    low_rows = [json.loads(l) for l in (CS_ROOT / LOWWIDTH / "lowwidth_rows.jsonl").read_text().splitlines()]
    sets = {"rcasr_bm25_w2": {r["image_id"]: r["rcasr_bm25_w2"] for r in low_rows},
            "rcasr_bm25_w3": rx.e2_sets(v2_rows)["rcasr_bm25"]}
    keep = ("gv_recall", "gv_precision_closed_world", "false_alarms_per_image", "need_review_rate")
    per_arm = {}
    for set_name, selected in sets.items():
        per_arm[set_name] = {}
        for arm, key in (("A_frozen_compliant", "A"), ("B_abstain_need_review", "B")):
            metrics = rx.e2e_metrics(selected, internal["labels"][key], statuses, family)
            per_arm[set_name][arm] = {k: metrics[k] for k in keep}
        pairs = {(i, r) for i, rs in selected.items() for r in rs}
        moved = [k for k in internal["rerouted_keys"] if k in pairs]
        per_arm[set_name]["rerouted_pairs"] = len(moved)
        per_arm[set_name]["rerouted_gold_status"] = dict(Counter(statuses[i].get(r, "unrecorded") for i, r in moved))
    public_review = {arm: sum(v in ("need_review", None) for v in public["labels"][key].values()) / public["n_pairs"]
                     for arm, key in (("A_frozen_compliant", "A"), ("B_abstain_need_review", "B"))}
    strip = lambda c: {k: v for k, v in c.items() if k not in ("labels", "rerouted_keys")}  # noqa: E731
    result = {"plan": PLAN, "snapshot_sha256": sha256(OUT / "input_snapshot.json"),
              "decision": {"checks_passed": passed,
                           "adopt": "B_abstain_need_review" if passed else "A_frozen_compliant",
                           "rule": "adopt B iff check 1 and check 2 hold exactly (prespec); cost is descriptive"},
              "internal_cache": strip(internal), "public_cache": strip(public),
              "internal_e2e": per_arm, "public_need_review_rate": public_review,
              "limits": ["gold-status counts of re-routed pairs are descriptive and development-exposed",
                         "need_review under B is an abstention for human review, not a verdict"]}
    write_json(OUT / "s34_results.json", result)
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
        print(json.dumps({k: result[k] for k in ("decision", "internal_e2e", "public_need_review_rate")}, indent=2))
        for name in ("internal_cache", "public_cache"):
            c = result[name]
            print(name, {k: c[k] for k in ("n_pairs", "check1_a_reproduces_cache", "check2_identical_non_compliant",
                                           "n_non_compliant", "routes", "rerouted")})


if __name__ == "__main__":
    main()
