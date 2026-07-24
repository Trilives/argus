#!/usr/bin/env python3
"""Bootstrap / extend a visual-screening rule library.

This is the authoring companion to the framework's central portability claim: a
new regulatory regime is onboarded by decomposing it into the same rule-unit
schema, not by retraining. The tool walks you through one rule at a time,
derives the fields that must stay in sync (``required_checkpoints`` mirrors the
keys of ``visual_checkpoints``), validates against ``rules_schema_en.json`` and a
few cross-field invariants the JSON schema cannot express, and writes the
library back atomically.

Usage
-----
    # Interactively add rules to the runtime library (creates it if absent):
    python scripts/bootstrap_rule_library.py

    # Start a brand-new library for another regime:
    python scripts/bootstrap_rule_library.py --path data/rules/osha_subpartM.json --new

    # Emit a blank, comment-annotated rule template (no prompts):
    python scripts/bootstrap_rule_library.py --template

    # Validate an existing library without editing it:
    python scripts/bootstrap_rule_library.py --check data/rules/rules_en.json

Runs under bare ``python3`` or ``uv run python``; ``jsonschema`` is used for full
schema validation when installed, otherwise a structural fallback runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES_PATH = Path(
    os.environ.get("CS_RULES_PATH") or (REPO_ROOT / "data" / "rules" / "rules_en.json")
)
SCHEMA_PATH = REPO_ROOT / "data" / "rules" / "rules_schema_en.json"

# Enum vocabularies mirrored from rules_schema_en.json so the prompts work even
# when the schema file is unavailable. Keep in sync with the schema.
MAJOR_CATEGORIES = (
    "Opening Hazard",
    "Edge Hazard",
    "Unsafe Behavior",
    "Civilized Construction Hazard",
)
EVIDENCE_TYPES = (
    "image",
    "video",
    "inspection_record",
    "sensor",
    "certificate_system",
    "permit",
    "site_plan",
    "text_log",
)
DECISION_SCOPES = (
    "visual_screening_only",
    "requires_external_evidence",
    "normative_decision_available",
)
PRIORITY_RE = re.compile(r"^P[0-9]+$")

# Canonical field order for a written rule (matches the schema's "required" list).
RULE_FIELD_ORDER = (
    "rule_id",
    "major_category",
    "subcategory",
    "rule_name",
    "priority",
    "evidence_types",
    "thresholds",
    "visual_checkpoints",
    "required_checkpoints",
    "critical_checkpoints",
    "machine_rule",
    "positive_keywords",
    "exclusion_keywords",
    "source_level",
    "source_quote",
    "rectification_advice",
    "normative_rule",
    "visual_screening_rule",
    "non_visual_fields",
    "decision_scope",
    "visual_retrieval_text",
    "review_prompt",
    "no_subject_default",
)


# --------------------------------------------------------------------------- #
# Prompt helpers
# --------------------------------------------------------------------------- #
def _ask(prompt: str, *, default: str | None = None) -> str:
    """Read one line, returning ``default`` on an empty answer."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            raw = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            raise SystemExit("\nAborted (EOF).")
        if raw:
            return raw
        if default is not None:
            return default
        print("  A value is required.")


def prompt_str(prompt: str, *, default: str | None = None, allow_empty: bool = False) -> str:
    if allow_empty:
        try:
            return input(f"{prompt} (optional): ").strip()
        except EOFError:
            raise SystemExit("\nAborted (EOF).")
    return _ask(prompt, default=default)


def prompt_validated(prompt: str, validate: Callable[[str], str | None], *, default: str | None = None) -> str:
    """Prompt until ``validate`` returns None (ok) instead of an error string."""
    while True:
        value = _ask(prompt, default=default)
        error = validate(value)
        if error is None:
            return value
        print(f"  {error}")


def prompt_choice(prompt: str, options: tuple[str, ...], *, default: str | None = None) -> str:
    print(f"{prompt}:")
    for i, opt in enumerate(options, 1):
        marker = "  (default)" if opt == default else ""
        print(f"  {i}) {opt}{marker}")
    while True:
        raw = input("  choose number or exact value" + (f" [{default}]" if default else "") + ": ").strip()
        if not raw and default is not None:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw in options:
            return raw
        print("  Invalid choice.")


def prompt_multi_choice(prompt: str, options: tuple[str, ...], *, default: list[str]) -> list[str]:
    print(f"{prompt} (comma-separated numbers or values)")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    raw = input(f"  [{', '.join(default)}]: ").strip()
    if not raw:
        return list(default)
    chosen: list[str] = []
    for token in (t.strip() for t in raw.split(",") if t.strip()):
        if token.isdigit() and 1 <= int(token) <= len(options):
            value = options[int(token) - 1]
        elif token in options:
            value = token
        else:
            print(f"  ignoring unknown value: {token}")
            continue
        if value not in chosen:
            chosen.append(value)
    return chosen or list(default)


def prompt_list(prompt: str) -> list[str]:
    """A comma-separated list; empty answer yields an empty list."""
    raw = input(f"{prompt} (comma-separated, optional): ").strip()
    if not raw:
        return []
    seen: list[str] = []
    for item in (i.strip() for i in raw.split(",")):
        if item and item not in seen:
            seen.append(item)
    return seen


def prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


# --------------------------------------------------------------------------- #
# Field builders
# --------------------------------------------------------------------------- #
def build_visual_checkpoints() -> dict[str, str]:
    """Collect checkpoint key -> instruction pairs.

    Zero checkpoints is allowed but only sensible for sensor-only
    (``requires_external_evidence``) rules, so an empty answer is confirmed.
    """
    print(
        "\nVisual checkpoints — the atomic things a reviewer reads off ONE image.\n"
        "Use snake_case keys, ideally ending in _present for subject/protection gates\n"
        "(e.g. horizontal_opening_present, guardrail_present). Blank key to finish."
    )
    checkpoints: dict[str, str] = {}
    while True:
        key = input(f"  checkpoint key #{len(checkpoints) + 1} (blank to finish): ").strip()
        if not key:
            if checkpoints:
                return checkpoints
            if prompt_yes_no("    No checkpoints — only valid for a sensor/external-evidence rule. Finish with none?", default=False):
                return checkpoints
            continue
        if key in checkpoints:
            print("    Duplicate key; overwriting its instruction.")
        instruction = _ask(f"    instruction for '{key}'")
        checkpoints[key] = instruction


def build_critical_checkpoints(required: list[str]) -> list[str]:
    print(
        "\nCritical checkpoints — the subset of the above that GATE the single-image\n"
        "verdict (if one is unobservable, the image alone cannot decide). May be empty\n"
        "for requires_external_evidence rules."
    )
    while True:
        chosen = prompt_list("  critical checkpoint keys")
        unknown = [c for c in chosen if c not in required]
        if unknown:
            print(f"    Not in this rule's checkpoints: {', '.join(unknown)}")
            continue
        return chosen


def build_thresholds() -> dict[str, float]:
    print("\nNumeric thresholds used by the machine/normative rule (blank name to finish; empty is fine).")
    thresholds: dict[str, float] = {}
    while True:
        name = input(f"  threshold name #{len(thresholds) + 1} (blank to finish): ").strip()
        if not name:
            return thresholds
        value = prompt_validated(
            f"    numeric value for '{name}'",
            lambda v: None if _is_number(v) else "Enter a number.",
        )
        thresholds[name] = float(value) if "." in value or "e" in value.lower() else int(value)


def build_non_visual_fields() -> dict[str, dict[str, Any]]:
    if not prompt_yes_no("\nAdd non-visual (external-evidence) fields?", default=False):
        return {}
    fields: dict[str, dict[str, Any]] = {}
    while True:
        name = input(f"  non-visual field key #{len(fields) + 1} (blank to finish): ").strip()
        if not name:
            return fields
        fields[name] = {
            "description": _ask(f"    description for '{name}'"),
            "source": prompt_list("    source(s) (e.g. manual_measurement, certificate_system)") or ["manual_review"],
            "required_for_normative_decision": prompt_yes_no("    required for a normative decision?", default=True),
            "use_in_current_experiment": prompt_yes_no("    used in the current experiment?", default=False),
        }


def build_rule(existing_ids: set[str]) -> dict[str, Any]:
    print("\n" + "=" * 70 + "\nNew rule\n" + "=" * 70)

    def _validate_id(value: str) -> str | None:
        if value in existing_ids:
            return "That rule_id already exists in this library."
        return None

    rule_id = prompt_validated(
        "rule_id (stable, e.g. R-OPN-001-horizontal-opening-protection)", _validate_id
    )
    major_category = prompt_choice("major_category", MAJOR_CATEGORIES, default=MAJOR_CATEGORIES[0])
    subcategory = _ask("subcategory (e.g. Horizontal opening)")
    rule_name = _ask("rule_name (the violation, stated plainly)")
    priority = prompt_validated(
        "priority (P0/P1/P2/...)",
        lambda v: None if PRIORITY_RE.match(v) else "Must look like P0, P1, P2 ...",
        default="P2",
    )
    evidence_types = prompt_multi_choice("evidence_types", EVIDENCE_TYPES, default=["image"])

    thresholds = build_thresholds()
    visual_checkpoints = build_visual_checkpoints()
    required_checkpoints = list(visual_checkpoints.keys())  # kept in sync by construction
    critical_checkpoints = build_critical_checkpoints(required_checkpoints)

    print(
        "\nDecision expressions — boolean/comparison expressions over the checkpoint\n"
        "keys and thresholds above (e.g. 'guardrail_present == no OR safety_net_present == no')."
    )
    visual_screening_rule = _ask("  visual_screening_rule (single-image risk expression)")
    machine_rule = prompt_str("  machine_rule (legacy expression)", default=visual_screening_rule)
    normative_rule = prompt_str("  normative_rule (full compliance expression)", default=visual_screening_rule)

    positive_keywords = prompt_list("\npositive_keywords (scene nouns that should retrieve this rule)")
    exclusion_keywords = prompt_list("exclusion_keywords (phrases meaning the rule does NOT apply)")

    print("\nProvenance and guidance text:")
    source_level = _ask("  source_level (e.g. national_standard, local_standard, engineering_derived)")
    source_quote = _ask("  source_quote (the governing clause text or a faithful translation)")
    rectification_advice = _ask("  rectification_advice (what to do to fix it)")
    visual_retrieval_text = _ask("  visual_retrieval_text (compliance-neutral description of visible cues)")
    review_prompt = _ask("  review_prompt (what a human should check when routed to review)")

    non_visual_fields = build_non_visual_fields()
    decision_scope = prompt_choice("\ndecision_scope", DECISION_SCOPES, default=DECISION_SCOPES[0])

    rule = {
        "rule_id": rule_id,
        "major_category": major_category,
        "subcategory": subcategory,
        "rule_name": rule_name,
        "priority": priority,
        "evidence_types": evidence_types,
        "thresholds": thresholds,
        "visual_checkpoints": visual_checkpoints,
        "required_checkpoints": required_checkpoints,
        "critical_checkpoints": critical_checkpoints,
        "machine_rule": machine_rule,
        "positive_keywords": positive_keywords,
        "exclusion_keywords": exclusion_keywords,
        "source_level": source_level,
        "source_quote": source_quote,
        "rectification_advice": rectification_advice,
        "normative_rule": normative_rule,
        "visual_screening_rule": visual_screening_rule,
        "non_visual_fields": non_visual_fields,
        "decision_scope": decision_scope,
        "visual_retrieval_text": visual_retrieval_text,
        "review_prompt": review_prompt,
        "no_subject_default": "compliant",  # only value the schema permits
    }
    return _ordered(rule)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def structural_errors(rule: dict[str, Any], existing_ids: set[str]) -> list[str]:
    """Cross-field invariants (some go beyond what JSON schema can express)."""
    errors: list[str] = []
    missing = [f for f in RULE_FIELD_ORDER if f not in rule]
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    extra = [f for f in rule if f not in RULE_FIELD_ORDER]
    if extra:
        errors.append(f"unexpected fields: {', '.join(extra)}")

    if rule.get("rule_id") in existing_ids:
        errors.append(f"duplicate rule_id: {rule.get('rule_id')}")
    if rule.get("major_category") not in MAJOR_CATEGORIES:
        errors.append(f"major_category not in enum: {rule.get('major_category')!r}")
    if not PRIORITY_RE.match(str(rule.get("priority", ""))):
        errors.append(f"priority must match P[0-9]+: {rule.get('priority')!r}")
    if rule.get("decision_scope") not in DECISION_SCOPES:
        errors.append(f"decision_scope not in enum: {rule.get('decision_scope')!r}")
    if rule.get("no_subject_default") != "compliant":
        errors.append("no_subject_default must be 'compliant'")

    checkpoints = rule.get("visual_checkpoints") or {}
    required = rule.get("required_checkpoints") or []
    if set(required) != set(checkpoints):
        errors.append("required_checkpoints must equal the keys of visual_checkpoints")
    # Zero visual checkpoints is valid (sensor-only requires_external_evidence rules,
    # e.g. TSP/noise exceedance), so it is not an error here.
    critical_extra = set(rule.get("critical_checkpoints") or []) - set(required)
    if critical_extra:
        errors.append(f"critical_checkpoints not in required: {', '.join(sorted(critical_extra))}")
    if not rule.get("evidence_types"):
        errors.append("evidence_types must have at least one entry")
    return errors


def schema_errors(rules: list[dict[str, Any]]) -> list[str]:
    """Full JSON-schema validation when ``jsonschema`` is importable."""
    try:
        import jsonschema  # type: ignore
    except ModuleNotFoundError:
        return []
    if not SCHEMA_PATH.exists():
        return []
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{list(err.path)}: {err.message}"
        for err in sorted(validator.iter_errors(rules), key=lambda e: list(e.path))
    ]


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_library(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path} does not contain a JSON array of rules.")
    return data


def save_library(path: Path, rules: list[dict[str, Any]]) -> None:
    """Atomic write: serialise to a temp file in the same dir, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".rules_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(rules, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _ordered(rule: dict[str, Any]) -> dict[str, Any]:
    return {field: rule[field] for field in RULE_FIELD_ORDER if field in rule}


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def blank_template() -> dict[str, Any]:
    """A schema-shaped skeleton with placeholder values, for offline editing."""
    return _ordered(
        {
            "rule_id": "R-XXX-000-short-slug",
            "major_category": MAJOR_CATEGORIES[0],
            "subcategory": "<subcategory>",
            "rule_name": "<the violation stated plainly>",
            "priority": "P2",
            "evidence_types": ["image"],
            "thresholds": {},
            "visual_checkpoints": {
                "subject_present": "Is the rule's subject visible? Judge only clear visible evidence.",
                "protection_present": "Is the required protection present and intact?",
            },
            "required_checkpoints": ["subject_present", "protection_present"],
            "critical_checkpoints": ["subject_present"],
            "machine_rule": "subject_present == yes AND protection_present == no",
            "positive_keywords": ["<scene noun>"],
            "exclusion_keywords": ["<phrase meaning rule does not apply>"],
            "source_level": "national_standard",
            "source_quote": "<governing clause text or faithful translation>",
            "rectification_advice": "<how to fix it>",
            "normative_rule": "subject_present == yes AND protection_present == no",
            "visual_screening_rule": "subject_present == yes AND protection_present == no",
            "non_visual_fields": {},
            "decision_scope": "visual_screening_only",
            "visual_retrieval_text": "<compliance-neutral description of visible cues>",
            "review_prompt": "<what a human should verify when routed to review>",
            "no_subject_default": "compliant",
        }
    )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_check(path: Path) -> int:
    rules = load_library(path)
    if not rules:
        print(f"{path}: empty or missing.")
        return 1
    problems: list[str] = []
    seen: set[str] = set()
    for i, rule in enumerate(rules):
        for err in structural_errors(rule, seen):
            problems.append(f"rule[{i}] ({rule.get('rule_id', '?')}): {err}")
        rid = rule.get("rule_id")
        if isinstance(rid, str):
            seen.add(rid)
    problems.extend(schema_errors(rules))
    if problems:
        print(f"{path}: {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"{path}: OK — {len(rules)} rules, {len(seen)} unique ids, schema valid.")
    return 0


def cmd_interactive(path: Path, *, start_new: bool, dry_run: bool) -> int:
    if start_new and path.exists():
        if not prompt_yes_no(f"{path} exists. Overwrite with a NEW library?", default=False):
            print("Aborted.")
            return 1
        rules: list[dict[str, Any]] = []
    else:
        rules = load_library(path)
    existing_ids = {r.get("rule_id") for r in rules if isinstance(r.get("rule_id"), str)}

    print(f"\nLibrary: {path}  ({len(rules)} existing rule(s))")
    added = 0
    while True:
        rule = build_rule(existing_ids)  # type: ignore[arg-type]
        errors = structural_errors(rule, existing_ids)  # type: ignore[arg-type]
        errors.extend(schema_errors(rules + [rule]))
        if errors:
            print("\nThis rule has problems:")
            for e in errors:
                print(f"  - {e}")
            if not prompt_yes_no("Keep it anyway (not recommended)?", default=False):
                if prompt_yes_no("Re-enter this rule from scratch?", default=True):
                    continue
                print("Discarded.")
                if prompt_yes_no("Add another rule?", default=False):
                    continue
                break
        print("\n" + json.dumps(rule, ensure_ascii=False, indent=2))
        if prompt_yes_no("Add this rule to the library?", default=True):
            rules.append(rule)
            existing_ids.add(rule["rule_id"])
            added += 1
        if not prompt_yes_no("Add another rule?", default=True):
            break

    if added == 0:
        print("\nNothing added.")
        return 0
    if dry_run:
        print(f"\n[dry-run] would write {len(rules)} rules to {path} (added {added}).")
        return 0
    save_library(path, rules)
    print(f"\nWrote {len(rules)} rules to {path} (added {added}).")
    print("Next: regenerate rule_assets and run bootstrap_rule_library.py --check on the file.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--path", type=Path, default=DEFAULT_RULES_PATH,
        help=f"Rule library JSON to create/extend (default: {DEFAULT_RULES_PATH}).",
    )
    parser.add_argument("--new", action="store_true", help="Start a new (empty) library at --path.")
    parser.add_argument("--template", action="store_true", help="Print a blank rule template and exit.")
    parser.add_argument("--check", type=Path, metavar="FILE", help="Validate an existing library and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Do everything except write the file.")
    args = parser.parse_args(argv)

    if args.template:
        print(json.dumps([blank_template()], ensure_ascii=False, indent=2))
        return 0
    if args.check is not None:
        return cmd_check(args.check)
    return cmd_interactive(args.path, start_new=args.new, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
