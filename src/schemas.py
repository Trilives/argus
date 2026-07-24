"""JSON output schemas for constrained decoding of the plain-generation stages.

Each schema mirrors the output shape its prompt file already asks for
(``Prompts_en/``); constrained decoding makes that shape a guarantee instead of
a request, eliminating the silent parse-failure degradation the TODO flags
(empty evidence, ``kept_ids or broad``). Schemas stay permissive — required
keys are only the ones a stage's parser actually reads, and extra keys are
allowed — so constraining can never reject content the pipeline would have
accepted.

Gated by ``config.GUIDED_JSON`` via :func:`guided`; the tool-calling agent
loops are unaffected (tool arguments are already schema-bound by vLLM).
"""

from __future__ import annotations

from typing import Any

import config

_STRING_LIST = {"type": "array", "items": {"type": "string"}}

_CHECKPOINT_EVIDENCE = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "checkpoint": {"type": "string"},
            "visible_evidence": {"type": "string"},
            "status": {"type": "string", "enum": ["satisfied", "violated", "not_visible", "need_review"]},
            "evidence_type": {
                "type": "string",
                "enum": ["direct_visible", "indirect_visible", "not_visible", "non_visual_required"],
            },
            "confidence": {"type": "number"},
        },
        "required": ["checkpoint", "status"],
    },
}

# Chain (Stage 3), J1, J2 and the J3 Stage-5 judgement share one verdict shape;
# only the checklist key differs and no parser requires it, so one schema serves
# all four. Required keys match what run_end_to_end.judge_image reads.
_VERDICT = {
    "type": "object",
    "properties": {
        "matched_rule": {
            "type": "object",
            "properties": {"rule_id": {"type": "string"}, "rule_name": {"type": "string"}},
            "required": ["rule_id"],
        },
        "visual_checklist_alignment": _CHECKPOINT_EVIDENCE,
        "missing_information": _STRING_LIST,
        "compliance_judgement": {
            "type": "object",
            "properties": {
                "compliance_label": {
                    "type": "string",
                    "enum": ["compliant", "non_compliant", "uncertain", "need_review"],
                },
                "reason": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["compliance_label"],
        },
        "rectification_suggestion": {"type": "object"},
        "evidence_chain_summary": {"type": "string"},
    },
    "required": ["matched_rule", "compliance_judgement"],
}

_EVIDENCE_RULE = {
    "type": "object",
    "properties": {
        "rule_id": {"type": "string"},
        "checkpoint_evidence": _CHECKPOINT_EVIDENCE,
        "missing_information": _STRING_LIST,
    },
    "required": ["rule_id", "checkpoint_evidence"],
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "facts_generic": {
        "type": "object",
        "properties": {"image_facts": _STRING_LIST, "unclear_or_missing": _STRING_LIST},
        "required": ["image_facts"],
    },
    "facts_scene": {
        "type": "object",
        "properties": {
            "scene_objects": _STRING_LIST,
            "spatial_relations": _STRING_LIST,
            "unclear_or_missing": _STRING_LIST,
        },
        "required": ["scene_objects"],
    },
    "evidence": _EVIDENCE_RULE,
    "evidence_batch": {
        "type": "object",
        "properties": {"rules": {"type": "array", "items": _EVIDENCE_RULE}},
        "required": ["rules"],
    },
    "verdict": _VERDICT,
    "aggregate": {
        "type": "object",
        "properties": {"aggregated_prompt": {"type": "string"}},
        "required": ["aggregated_prompt"],
    },
    "observations": {
        "type": "object",
        "properties": {"observations": {"type": "array", "items": {"type": "object"}}},
        "required": ["observations"],
    },
    "filter": {
        "type": "object",
        "properties": {"kept_rule_ids": _STRING_LIST},
        "required": ["kept_rule_ids"],
    },
}


def guided(name: str) -> dict[str, Any] | None:
    """Schema for ``name`` when constrained decoding is on, else ``None``."""
    return SCHEMAS[name] if config.GUIDED_JSON else None
