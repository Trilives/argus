"""Offline diagnostic orchestration and figures; deliberately no RCASR win claim."""
from __future__ import annotations

import json
from pathlib import Path

from research_snapshot import read_json
from retrieval.base import facts_query
from retrieval.bm25 import BM25Retriever
from set_selection import ALPHAS, FIXED_K, FIXED_THRESHOLDS, calibrate, matched_rank_sets, select
from set_selection_metrics import metrics, paired_intervals


def load_inputs(root: Path) -> tuple[dict, dict, dict, list]:
    gold = read_json(root / "data/annotations/image_rule_gold.json")["gold"]
    facts = read_json(root / "results/retrieval/gold_facts_generic_en.json")
    index = read_json(root / "data/rule_assets/rule_index.json")
    rules = read_json(root / "data/rules/rules_en.json")
    family = {r["rule_id"]: r["rule_id"].split("-")[1] for r in rules}
    if {r["rule_id"] for r in index} != set(family) or set(gold) != set(facts):
        raise ValueError("facts/index/gold populations do not match frozen inputs")
    retriever = BM25Retriever(index)
    scores = {}
    for image_id in sorted(gold):
        row = facts[image_id]
        if row.get("parse_error") or not row.get("image_facts"):
            raise ValueError(f"missing/failed fact cache for {image_id}; do not silently exclude")
        ranked = retriever.retrieve(facts_query(row["image_facts"]), top_k=len(index))
        scores[image_id] = {r.rule_id: r.score / (1 + r.score) for r in ranked}
    positives = {i: {r for r, s in row["rule_statuses"].items() if s == "non_compliant"}
                 for i, row in gold.items()}
    clusters = read_json(root / "results/data_audit/near_duplicates.json")["clusters"]
    return scores, positives, family, clusters


def evaluate(root: Path) -> tuple[dict, dict]:
    scores, positives, families, clusters = load_inputs(root)
    arms = {}
    selected_rows = {}
    for k in FIXED_K:
        chosen = {i: select(s, 0.)[:k] for i, s in scores.items()}
        arms[f"fixed_k_{k}"] = {"metrics": metrics(chosen, positives, families)}
        selected_rows[f"fixed_k_{k}"] = chosen
    for threshold in FIXED_THRESHOLDS:
        name = f"fixed_threshold_{threshold}"
        chosen = {i: select(s, threshold) for i, s in scores.items()}
        arms[name] = {"threshold": threshold, "metrics": metrics(chosen, positives, families)}
        selected_rows[name] = chosen
    for alpha in ALPHAS:
        fit = calibrate(scores, positives, alpha)
        name = f"crc_{alpha}"
        if fit["threshold"] is None:
            arms[name] = {"calibration": fit, "status": "need_review"}
            continue
        chosen = {i: select(s, fit["threshold"]) for i, s in scores.items()}
        m = metrics(chosen, positives, families)
        matched = matched_rank_sets(scores, m["width_total"])
        mm = metrics(matched, positives, families)
        arms[name] = {"calibration": fit, "metrics": m, "matched_rank_metrics": mm,
                      "recall_delta": m["recall_micro"] - mm["recall_micro"],
                      "risk_delta": m["risk"] - mm["risk"],
                      "paired_intervals": paired_intervals(chosen, matched, positives, clusters),
                      "mean_width_above_three": m["width_mean"] > 3}
        selected_rows[name], selected_rows[f"{name}_matched_rank"] = chosen, matched
    result = {"scope": "all-500 resubstitution diagnostic; not E1 or independent risk validation",
              "method": "BM25 unstructured score baseline, not RCASR",
              "n_new_model_calls": 0, "n_new_labels": 0,
              "cost_scope": "42 rules scored/image offline; historical fact extraction cost excluded",
              "family_definition": "BHV/CIV/EDG/OPN rule-ID categories, not a learned hierarchy",
              "cap_semantics": "over-3 sets deferred in full; reported recall is pre-cap selection recall",
              "uncertainty": "conditional duplicate-cluster bootstrap; not site-clustered",
              "arms": arms}
    rows = {i: {"scores": scores[i], "selected": {a: r[i] for a, r in selected_rows.items()}}
            for i in scores}
    return result, rows


def render(result: dict) -> str:
    lines = ["# Existing-label set-selection diagnostics", "", result["scope"], "",
             "Unstructured BM25 baseline only. Thresholds are fitted and diagnosed on the same 500",
             "development-exposed images. No new inference, annotations or site partitions.", "",
             "Recall is selection recall before candidate-cap deferral. Width is not false exposure.",
             "The review rate counts images with width >3; it does not measure human time saved.", "",
             "| arm | recall | image miss risk | width mean | width p95 | cap review rate |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, arm in result["arms"].items():
        m = arm.get("metrics")
        if m:
            lines.append(f"| {name} | {m['recall_micro']:.4f} | {m['risk']:.4f} | "
                         f"{m['width_mean']:.3f} | {m['width_p95']:.1f} | {m['cap_review_rate']:.3f} |")
    lines += ["", "## Exact matched-width rank mixtures", "",
              "All comparisons exploratory; paired 95% descriptive duplicate-cluster intervals.",
              "They condition on fitted thresholds and do not account for site dependence or fitting.", "",
              "| alpha | threshold | recall delta | interval | mean width >3 |",
              "|---|---:|---:|---|---|"]
    for arm in result["arms"].values():
        if "paired_intervals" not in arm:
            continue
        ci = arm["paired_intervals"]["recall_delta_ci"]
        lines.append(f"| {arm['calibration']['alpha']} | {arm['calibration']['threshold']:.3f} | "
                     f"{arm['recall_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                     f"{arm['mean_width_above_three']} |")
    return "\n".join(lines) + "\n"


def plot(result: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = [result["arms"][f"fixed_k_{k}"]["metrics"] for k in FIXED_K]
    crc = [result["arms"][f"crc_{a}"] for a in ALPHAS if "metrics" in result["arms"][f"crc_{a}"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    axes[0].plot([r["width_mean"] for r in fixed], [r["recall_micro"] for r in fixed], "o-", label="Fixed k")
    axes[0].plot([r["metrics"]["width_mean"] for r in crc], [r["metrics"]["recall_micro"] for r in crc], "s--", label="Unstructured CRC")
    axes[0].axvline(3, color="gray", linestyle=":", label="Candidate cap")
    axes[0].set(xlabel="Mean candidate width (before deferral)", ylabel="Recorded-violation recall")
    axes[0].legend()
    nominal = [r["calibration"]["alpha"] for r in crc]
    empirical = [r["metrics"]["risk"] for r in crc]
    bounds = [r["paired_intervals"]["left_risk_ci"] for r in crc]
    axes[1].plot(nominal, nominal, ":", color="gray", label="Nominal target")
    axes[1].plot(nominal, empirical, "o-", label="Same-data diagnostic")
    axes[1].fill_between(nominal, [b[0] for b in bounds], [b[1] for b in bounds], alpha=.2,
                         label="Conditional 95% interval")
    axes[1].set(xlabel="Nominal alpha", ylabel="Per-image observed miss risk")
    axes[1].legend()
    fig.suptitle("Retrospective resubstitution diagnostics — no independent calibration guarantee")
    fig.savefig(out / "selection_diagnostics.svg")
    fig.savefig(out / "selection_diagnostics.png", dpi=180)
    plt.close(fig)
