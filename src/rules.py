"""Rule extraction and indexing for the CS evidence-chain experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import paths
from io_utils import load_json, write_json

RULE_UNIT_FIELDS = (
    "rule_id",
    "major_category",
    "subcategory",
    "rule_name",
    "priority",
    "source_level",
    "source_quote",
    "thresholds",
    "visual_checkpoints",
    "visual_screening_rule",
    "visual_retrieval_text",
    "non_visual_fields",
    "decision_scope",
    "review_prompt",
    "rectification_advice",
    "positive_keywords",
    "exclusion_keywords",
)


@dataclass(frozen=True)
class RuleAssetPaths:
    rule_units: str
    rule_index: str
    rule_stats: str
    rule_fields: str


def load_rules() -> list[dict[str, Any]]:
    """Load the CS-local rule library (copied from the shared canonical source)."""
    rules = load_json(paths.RULES_PATH)
    if not isinstance(rules, list):
        raise ValueError(f"Expected list of rules in {paths.RULES_PATH}")
    return [dict(rule) for rule in rules]


def load_rule_units() -> list[dict[str, Any]]:
    rules = load_json(paths.RULE_UNITS_PATH)
    if not isinstance(rules, list):
        raise ValueError(f"Expected list of rule units in {paths.RULE_UNITS_PATH}")
    return [dict(rule) for rule in rules]


def rule_by_id(rules: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {rule["rule_id"]: rule for rule in rules}


def extract_rule_unit(rule: dict[str, Any]) -> dict[str, Any]:
    return {field: rule.get(field) for field in RULE_UNIT_FIELDS}


def checkpoint_text(rule: dict[str, Any]) -> str:
    checkpoints = rule.get("visual_checkpoints") or {}
    if isinstance(checkpoints, dict):
        return "; ".join(str(value) for value in checkpoints.values())
    if isinstance(checkpoints, list):
        return "; ".join(str(value) for value in checkpoints)
    return str(checkpoints)


def keyword_text(rule: dict[str, Any], field: str) -> str:
    values = rule.get(field) or []
    if isinstance(values, list):
        return "; ".join(str(value) for value in values)
    return str(values)


def derived_scene_text(rule: dict[str, Any]) -> str:
    parts = [
        rule.get("major_category", ""),
        rule.get("subcategory", ""),
        rule.get("rule_name", ""),
        checkpoint_text(rule),
        keyword_text(rule, "positive_keywords"),
        keyword_text(rule, "exclusion_keywords"),
    ]
    return ". ".join(part for part in parts if part)


def build_rule_index(rule_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for rule in rule_units:
        derived_text = derived_scene_text(rule)
        retrieval_text = str(rule.get("visual_retrieval_text") or "")
        source_text = str(rule.get("source_quote") or "")
        rectification_text = str(rule.get("rectification_advice") or "")
        index.append(
            {
                "rule_id": rule["rule_id"],
                "major_category": rule.get("major_category"),
                "subcategory": rule.get("subcategory"),
                "rule_name": rule.get("rule_name"),
                "decision_scope": rule.get("decision_scope"),
                "priority": rule.get("priority"),
                "derived_scene_text": derived_text,
                "visual_retrieval_text": retrieval_text,
                "evidence_chain_text": ". ".join(
                    part
                    for part in (
                        derived_text,
                        retrieval_text,
                        source_text,
                        rectification_text,
                    )
                    if part
                ),
            }
        )
    return index


def build_rule_stats(rule_units: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rule_count": len(rule_units),
        "major_category": dict(Counter(rule.get("major_category") for rule in rule_units)),
        "decision_scope": dict(Counter(rule.get("decision_scope") for rule in rule_units)),
        "priority": dict(Counter(rule.get("priority") for rule in rule_units)),
        "fields": list(RULE_UNIT_FIELDS),
    }


def rule_fields_markdown() -> str:
    rows = [
        ("rule_id", "Unique rule identifier, used for traceability in evidence chains and evaluation."),
        ("major_category", "Top-level defect category."),
        ("subcategory", "Second-level scene or risk-object category."),
        ("rule_name", "Human- and paper-facing rule name."),
        ("priority", "Risk priority."),
        ("source_level", "Provenance level of the rule."),
        ("source_quote", "Summary of the normative basis, cited in the evidence chain."),
        ("thresholds", "Thresholds involved in a full normative decision; the VLM is not asked to guess them."),
        ("visual_checkpoints", "Checkpoints decidable from a single image or weakly visual."),
        ("visual_screening_rule", "The rule logic used by the current CS single-image visual screening."),
        ("visual_retrieval_text", "Rule description oriented toward image-text retrieval."),
        ("non_visual_fields", "Information requiring measurement, records, certificates, or sensors."),
        ("decision_scope", "Visual screening, image-decidable, or requires external evidence."),
        ("review_prompt", "Information to prompt a human reviewer for when evidence is insufficient."),
        ("rectification_advice", "Normatively grounded rectification advice."),
        ("positive_keywords", "Violation / risk-side keywords, aiding diagnosis and text retrieval."),
        ("exclusion_keywords", "Compliant / exclusion-side keywords, aiding retrieval analysis."),
    ]
    lines = [
        "# CS evidence-chain rule fields",
        "",
        "This directory holds the CS line's experimental rule assets derived from the rule library. "
        "The rules come from `CS/data/rules/rules_en.json` as the local runtime source (the repo is "
        "self-contained and no longer depends on an external `shared/`); this document describes the "
        "fields currently used by the CS evidence-chain experiments.",
        "",
        "| Field | Purpose |",
        "|---|---|",
    ]
    lines.extend(f"| `{field}` | {description} |" for field, description in rows)
    lines.extend(
        [
            "",
            "Note: the CS main experiment defaults to single-image visual screening; a full normative "
            "decision involving `thresholds` and `non_visual_fields` should emit review items rather "
            "than have the VLM fabricate values it cannot see.",
            "",
        ]
    )
    return "\n".join(lines)


def build_rule_assets() -> RuleAssetPaths:
    rule_units = [extract_rule_unit(rule) for rule in load_rules()]
    rule_index = build_rule_index(rule_units)
    rule_stats = build_rule_stats(rule_units)

    write_json(paths.RULE_UNITS_PATH, rule_units)
    write_json(paths.RULE_INDEX_PATH, rule_index)
    write_json(paths.RULE_STATS_PATH, rule_stats)
    paths.RULE_FIELDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    paths.RULE_FIELDS_PATH.write_text(rule_fields_markdown(), encoding="utf-8")

    return RuleAssetPaths(
        rule_units=str(paths.RULE_UNITS_PATH),
        rule_index=str(paths.RULE_INDEX_PATH),
        rule_stats=str(paths.RULE_STATS_PATH),
        rule_fields=str(paths.RULE_FIELDS_PATH),
    )
