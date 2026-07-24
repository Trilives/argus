"""Lightweight validation for evidence-chain outputs.

This module deliberately avoids a jsonschema dependency; it checks the required
top-level shape that matters for pipeline smoke tests. Full schema validation can
be added later if the project adds jsonschema to pyproject.toml.
"""

from __future__ import annotations

from typing import Any

import paths
from io_utils import load_json


def required_schema_fields() -> list[str]:
    schema = load_json(paths.EVIDENCE_CHAIN_SCHEMA_PATH)
    required = schema.get("required", []) if isinstance(schema, dict) else []
    return [str(field) for field in required]


def validate_chain_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in required_schema_fields():
        if field not in record:
            errors.append(f"missing required field: {field}")

    checklist = record.get("visual_checklist_alignment")
    if checklist is not None and not isinstance(checklist, list):
        errors.append("visual_checklist_alignment must be a list")

    judgement = record.get("compliance_judgement")
    if judgement is not None and not isinstance(judgement, dict):
        errors.append("compliance_judgement must be an object")

    rectification = record.get("rectification_suggestion")
    if rectification is not None and not isinstance(rectification, dict):
        errors.append("rectification_suggestion must be an object")

    return errors


def validate_chain_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        errors = validate_chain_record(record)
        rows.append(
            {
                "sample_id": record.get("sample_id"),
                "image_id": record.get("image_id"),
                "valid": not errors,
                "errors": errors,
            }
        )
    return {
        "record_count": len(records),
        "valid_count": sum(1 for row in rows if row["valid"]),
        "invalid_count": sum(1 for row in rows if not row["valid"]),
        "rows": rows,
    }
