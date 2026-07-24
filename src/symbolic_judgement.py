"""J3-sym: symbolic verdicts over Stage-4 checkpoint evidence.

Every rule's ``visual_screening_rule`` is a machine-evaluable boolean formula
over its checkpoint ids (``subject_present == yes AND (defect conditions)``).
Stage 4 reports a per-checkpoint status; this module evaluates the formula in
code with Kleene three-valued logic and replaces the Stage-5 LLM verdict:

- formula **true**  -> ``non_compliant`` (violation confirmed on evidence)
- formula **false** -> ``compliant`` (includes subject-absent: the leading
  ``== yes`` gate fails, so defect unknowns cannot flip the outcome)
- formula **unknown** because the *subject gate* is unresolved -> the rule's
  ``no_subject_default`` (uniformly ``compliant`` in the current library) —
  the absent-subject rejection the family-expansion arm needs
- otherwise **unknown** -> ``need_review`` (abstention channel)

The VLM only perceives; the formula decides. No model call happens here.
"""
from __future__ import annotations

import re
from typing import Any

from io_utils import load_json
import paths

STATUS_TO_VALUE = {"satisfied": True, "violated": False}
# status -> the checkpoint's three-valued answer; not_visible / need_review -> None

_TOKEN_RE = re.compile(r"\(|\)|==|\bAND\b|\bOR\b|\byes\b|\bno\b|[A-Za-z_][A-Za-z0-9_]*")
_ATOM_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(yes|no)")
# some rules (certification, weather, sensor readings) cannot be screened from
# a single image; their formula is this sentinel and they always abstain
NOT_EVALUABLE = "not_available_for_single_image"


class FormulaError(ValueError):
    """Raised when a visual_screening_rule does not parse."""


def formula_atoms(formula: str) -> list[str]:
    """Checkpoint ids referenced by the formula, in order of appearance."""
    return [m.group(1) for m in _ATOM_RE.finditer(formula)]


def parse_formula(formula: str):
    """Recursive-descent parse into ('or'|'and', [children]) / ('atom', id, expected)."""
    tokens = _TOKEN_RE.findall(formula)
    if not tokens:
        raise FormulaError(f"empty formula: {formula!r}")
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def take(expected: str | None = None) -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise FormulaError(f"unexpected end of formula: {formula!r}")
        tok = tokens[pos]
        if expected is not None and tok != expected:
            raise FormulaError(f"expected {expected!r} got {tok!r} in {formula!r}")
        pos += 1
        return tok

    def parse_expr():
        node = parse_term()
        children = [node]
        while peek() == "OR":
            take("OR")
            children.append(parse_term())
        return children[0] if len(children) == 1 else ("or", children)

    def parse_term():
        node = parse_factor()
        children = [node]
        while peek() == "AND":
            take("AND")
            children.append(parse_factor())
        return children[0] if len(children) == 1 else ("and", children)

    def parse_factor():
        if peek() == "(":
            take("(")
            node = parse_expr()
            take(")")
            return node
        name = take()
        if name in {"AND", "OR", ")", "==", "yes", "no"}:
            raise FormulaError(f"unexpected token {name!r} in {formula!r}")
        take("==")
        expected = take()
        if expected not in {"yes", "no"}:
            raise FormulaError(f"atom {name!r} compared to {expected!r} in {formula!r}")
        return ("atom", name, expected == "yes")

    tree = parse_expr()
    if pos != len(tokens):
        raise FormulaError(f"trailing tokens {tokens[pos:]} in {formula!r}")
    return tree


def _eval_node(node, values: dict[str, bool | None]) -> bool | None:
    """Kleene three-valued evaluation (None = unknown)."""
    kind = node[0]
    if kind == "atom":
        _, name, expect_yes = node
        value = values.get(name)
        if value is None:
            return None
        return value is expect_yes
    results = [_eval_node(child, values) for child in node[1]]
    if kind == "and":
        if any(r is False for r in results):
            return False
        return None if any(r is None for r in results) else True
    if any(r is True for r in results):
        return True
    return None if any(r is None for r in results) else False


def eval_formula(formula: str, statuses: dict[str, str]) -> bool | None:
    """Evaluate against ``{checkpoint_id: satisfied|violated|not_visible|need_review}``."""
    values = {k: STATUS_TO_VALUE.get(v) for k, v in statuses.items()}
    return _eval_node(parse_formula(formula), values)


def _rule_defaults() -> dict[str, str]:
    return {
        r["rule_id"]: r.get("no_subject_default", "compliant")
        for r in load_json(paths.RULES_PATH)
    }


_DEFAULTS: dict[str, str] | None = None


def symbolic_verdict(rule_evidence: dict[str, Any]) -> dict[str, Any]:
    """One rule's Stage-4 evidence item -> a judgement-shaped symbolic verdict."""
    global _DEFAULTS
    if _DEFAULTS is None:
        _DEFAULTS = _rule_defaults()

    rule_id = rule_evidence["rule_id"]
    formula = rule_evidence.get("visual_screening_rule") or ""
    if formula.strip() == NOT_EVALUABLE:
        return {
            "matched_rule": {"rule_id": rule_id, "rule_name": rule_evidence.get("rule_name")},
            "visual_decidability": {
                "label": "not_fully_decidable",
                "reason": "rule is not visually evaluable from a single image",
            },
            "compliance_judgement": {"compliance_label": "need_review", "confidence": None},
            "symbolic": {"route": "not_visually_evaluable", "formula_outcome": "unknown",
                         "unknown_atoms": [], "gate_atom": None},
            "checkpoint_evidence": rule_evidence.get("checkpoint_evidence", []),
            "missing_information": rule_evidence.get("missing_information", []),
        }
    statuses: dict[str, str] = {}
    confidences: list[float] = []
    for item in rule_evidence.get("checkpoint_evidence", []):
        if isinstance(item, dict) and item.get("checkpoint"):
            statuses[item["checkpoint"]] = item.get("status", "need_review")
            if isinstance(item.get("confidence"), (int, float)):
                confidences.append(float(item["confidence"]))

    outcome = eval_formula(formula, statuses)
    atoms = formula_atoms(formula)
    gate = atoms[0] if atoms else None
    unknown_atoms = [
        a for a in atoms if STATUS_TO_VALUE.get(statuses.get(a, "need_review")) is None
    ]

    if outcome is True:
        label, route = "non_compliant", "formula_true"
    elif outcome is False:
        label, route = "compliant", "formula_false"
    elif gate in unknown_atoms:
        label, route = _DEFAULTS.get(rule_id, "compliant"), "no_subject_default"
    else:
        label, route = "need_review", "unknown_needs_review"

    confidence = round(min(confidences), 3) if confidences else None
    return {
        "matched_rule": {"rule_id": rule_id, "rule_name": rule_evidence.get("rule_name")},
        "visual_decidability": {
            "label": "decidable" if outcome is not None else "not_fully_decidable",
            "reason": f"symbolic evaluation route: {route}",
        },
        "compliance_judgement": {"compliance_label": label, "confidence": confidence},
        "symbolic": {
            "route": route,
            "formula_outcome": {True: "true", False: "false", None: "unknown"}[outcome],
            "unknown_atoms": unknown_atoms,
            "gate_atom": gate,
        },
        "checkpoint_evidence": rule_evidence.get("checkpoint_evidence", []),
        "missing_information": rule_evidence.get("missing_information", []),
    }
