"""Evaluation helpers for CS rule retrieval and evidence-chain experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from retrieval import RetrievedRule


def topk_hit(retrieved: list[RetrievedRule], target_rule_id: str, k: int) -> bool:
    return any(item.rule_id == target_rule_id for item in retrieved[:k])


def category_hit(
    retrieved: list[RetrievedRule],
    target: dict[str, Any],
    *,
    k: int,
    level: str = "major_category",
) -> bool:
    expected = target.get(level)
    if expected is None:
        return False
    return any(getattr(item, level) == expected for item in retrieved[:k])


def summarize_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "sample_count": 0,
            "top1_rule_match": 0.0,
            "top3_rule_recall": 0.0,
            "top5_rule_recall": 0.0,
            "top1_major_category_match": 0.0,
        }
    return {
        "sample_count": total,
        "top1_rule_match": sum(row["hit_top1"] for row in rows) / total,
        "top3_rule_recall": sum(row["hit_top3"] for row in rows) / total,
        "top5_rule_recall": sum(row["hit_top5"] for row in rows) / total,
        "top1_major_category_match": sum(row["category_hit_top1"] for row in rows) / total,
    }


# --- Judgement-layer scoring (image_rule_gold.json rule_statuses) ------------
#
# The gold carries three states (see docs/EXPERIMENT_PROTOCOL.md §2); the model
# emits four (compliant | non_compliant | uncertain | need_review). Scoring maps
# the model output into gold space, then reports both the 3-way confusion matrix
# and the collapsed two-state detector view (positive class = non_compliant).

GOLD_STATUSES = ("compliant", "non_compliant", "undetermined")

# model compliance_label -> gold status space
MODEL_TO_GOLD = {
    "compliant": "compliant",
    "non_compliant": "non_compliant",
    "uncertain": "undetermined",
    "need_review": "undetermined",
}


def to_gold_status(model_label: str | None) -> str:
    """Map a model ``compliance_label`` into gold space.

    Both abstention labels collapse to ``undetermined``; anything unrecognised
    (including ``None`` / unparsed output) is treated as an abstention rather
    than a silent verdict.
    """
    return MODEL_TO_GOLD.get(model_label or "", "undetermined")


def to_detection_flag(status: str) -> str:
    """Collapse a gold-space status to the two-state detector view.

    Only a confirmed violation is ``flagged``; ``compliant`` and ``undetermined``
    are both ``not_flagged`` — i.e. no violation is reported from this view.
    """
    return "flagged" if status == "non_compliant" else "not_flagged"


def summarize_judgement(pairs: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
    """Score ``(gold_status, model_compliance_label)`` pairs.

    Returns the 3-way confusion matrix, the collapsed non-compliant detector
    metrics (precision / recall / F1), and selective-prediction coverage that
    treats ``undetermined`` as abstention.
    """
    total = len(pairs)
    empty_matrix = {g: {p: 0 for p in GOLD_STATUSES} for g in GOLD_STATUSES}
    if total == 0:
        return {
            "sample_count": 0,
            "confusion": empty_matrix,
            "detection": {
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            },
            "selective": {"coverage": 0.0, "committed": 0, "committed_accuracy": 0.0},
        }

    confusion = {g: {p: 0 for p in GOLD_STATUSES} for g in GOLD_STATUSES}
    tp = fp = fn = tn = 0
    committed = correct_committed = 0

    for gold_status, model_label in pairs:
        if gold_status not in GOLD_STATUSES:
            raise ValueError(f"unknown gold status: {gold_status!r}")
        pred = to_gold_status(model_label)
        confusion[gold_status][pred] += 1

        # two-state detector view (positive class = non_compliant)
        gold_flag = to_detection_flag(gold_status)
        pred_flag = to_detection_flag(pred)
        if gold_flag == "flagged" and pred_flag == "flagged":
            tp += 1
        elif gold_flag == "not_flagged" and pred_flag == "flagged":
            fp += 1
        elif gold_flag == "flagged" and pred_flag == "not_flagged":
            fn += 1
        else:
            tn += 1

        # selective view: undetermined == abstention; committing = a hard verdict
        if pred != "undetermined":
            committed += 1
            if pred == gold_status:
                correct_committed += 1

    def ratio(num: float, den: float) -> float:
        return num / den if den else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * precision * recall, precision + recall)

    return {
        "sample_count": total,
        "confusion": confusion,
        "detection": {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "selective": {
            "coverage": ratio(committed, total),
            "committed": committed,
            "committed_accuracy": ratio(correct_committed, committed),
        },
    }
