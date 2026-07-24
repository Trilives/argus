#!/usr/bin/env python3
"""Grounded-Violation (GV) scorer — reproduce the paper's headline metric.

This is a self-contained port of the metric in the research repository
(`experiments/pipeline/eval_grounded_violations.py`). Given the released
label-only gold and a label-only predictions file, it recomputes GV
precision/recall/F1 with no images or scene text — so a reviewer can verify the
reported numbers directly.

    python score_grounded_violations.py gold_labels.json predictions/<system>.jsonl

Gold (`gold_labels.json`): {image_stem: {rule_id: status}} where status is one of
compliant / non_compliant / undetermined. A rule is a "gold violation" iff its
status is non_compliant.

Predictions (JSONL, one object per image), label-only, either schema:
    {"image_id": "...", "model_label": "non_compliant", "matched_rule_id": "R-..."}
    {"image_id": "...", "violation_instances": [{"rule_id": "R-...", "compliance_label": "non_compliant"}]}

A predicted violation is correct iff its rule_id is in that image's gold
violated set (clause + status must both match; spatial location is out of scope).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

VIOLATION = "non_compliant"


def load_gold(path: Path) -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """image_stem -> (applicable rule set, violated rule set)."""
    statuses = json.loads(path.read_text(encoding="utf-8"))
    gold: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for stem, rule_statuses in statuses.items():
        applicable = frozenset(rule_statuses)
        violated = frozenset(r for r, s in rule_statuses.items() if s == VIOLATION)
        gold[Path(stem).stem] = (applicable, violated)
    return gold


def claimed_violations(rec: dict) -> set[str]:
    instances = rec.get("violation_instances")
    if isinstance(instances, list):
        return {
            i["rule_id"] for i in instances
            if isinstance(i, dict) and i.get("rule_id") and i.get("compliance_label") == VIOLATION
        }
    if rec.get("model_label") == VIOLATION and rec.get("matched_rule_id"):
        return {rec["matched_rule_id"]}
    return set()


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score(gold_path: Path, pred_path: Path) -> dict:
    gold = load_gold(gold_path)
    records = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    tp = pred = gold_count = 0
    macro_p: list[float] = []
    macro_r: list[float] = []
    for rec in records:
        stem = Path(str(rec["image_id"])).stem
        entry = gold.get(stem)
        if entry is None:
            continue
        _applicable, violated = entry
        claims = claimed_violations(rec)
        hit = len(claims & violated)
        tp += hit
        pred += len(claims)
        gold_count += len(violated)
        if claims:
            macro_p.append(hit / len(claims))
        if violated:
            macro_r.append(hit / len(violated))

    p_micro = tp / pred if pred else 0.0
    r_micro = tp / gold_count if gold_count else 0.0
    return {
        "n_images": len(records),
        "gv_tp": tp, "gv_pred": pred, "gv_gold": gold_count,
        "gv_precision_micro": round(p_micro, 4),
        "gv_recall_micro": round(r_micro, 4),
        "gv_f1_micro": round(f1(p_micro, r_micro), 4),
        "gv_precision_macro": round(mean(macro_p), 4),
        "gv_recall_macro": round(mean(macro_r), 4),
    }


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} gold_labels.json predictions.jsonl")
    result = score(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
