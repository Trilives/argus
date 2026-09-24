"""Applicability gate: decide whether a candidate provision governs this image at all.

The end-to-end evidence says grounded precision falls as the candidate set widens,
and the dominant failure is flagging provisions whose *subject* is not in the
photograph — a rule about floor openings fired at an image with no opening. The
judge is asked "is this compliant?" when the prior question, "does this provision
apply here?", was never asked.

This module makes that question explicit and, crucially, derives it from the frozen
library rather than from a hand-written list of subjects:

  A rule's ``visual_screening_rule`` is a Kleene formula over its checkpoints.
  Bind only the **positive** atoms (those compared ``== yes``) and leave every
  defect atom unknown. Kleene AND/OR are monotone in unknowns, so if the formula
  already evaluates to **False** under that partial binding, no assignment of the
  remaining checkpoints can make it true: the rule *cannot* be violated in this
  image. That is exactly inapplicability, and it is a property of the formula, not
  an opinion.

Positive atoms carry the rule's subject ("horizontal_opening_present",
"fall_risk_present"); negative atoms are protection and defect conditions
("cover_present == no"), whose falsity is a compliance finding rather than an
applicability one. So the gate asks the model only about the positive atoms, which
is both cheaper than a full checkpoint read and narrower to answer.

Three outcomes feed the pipeline:
  * **False**   -> ``not_applicable``: filtered, no violation claim, no further calls.
  * **True**    -> ``applicable``: proceed to the full evidence read and verdict.
  * **unknown** -> ``unknown``: the subject could not be confirmed or refuted, which
    routes to ``need_review`` — a structural abstention channel, not a scalar
    threshold.
"""

from __future__ import annotations

import re
from typing import Any

from symbolic_judgement import (
    NOT_EVALUABLE,
    STATUS_TO_VALUE,
    FormulaError,
    _eval_node,
    parse_formula,
)

_POSITIVE_ATOM_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*==\s*yes")

APPLICABLE = "applicable"
NOT_APPLICABLE = "not_applicable"
UNKNOWN = "unknown"


def positive_atoms(formula: str) -> list[str]:
    """Checkpoint ids the formula requires to be *present*, in order, deduplicated."""
    seen: set[str] = set()
    atoms: list[str] = []
    for match in _POSITIVE_ATOM_RE.finditer(formula or ""):
        atom = match.group(1)
        if atom not in seen:
            seen.add(atom)
            atoms.append(atom)
    return atoms


def gate_outcome(formula: str, statuses: dict[str, str]) -> bool | None:
    """Evaluate the formula with only the positive atoms bound, defects unknown.

    ``statuses`` maps checkpoint id -> satisfied/violated/not_visible/need_review,
    as reported for the positive atoms alone. Any atom not present, and every
    negative atom, is treated as unknown.
    """
    if not formula or formula.strip() == NOT_EVALUABLE:
        return None
    allowed = set(positive_atoms(formula))
    values: dict[str, bool | None] = {
        atom: STATUS_TO_VALUE.get(statuses.get(atom, "need_review"))
        for atom in allowed
    }
    try:
        return _eval_node(parse_formula(formula), values)
    except FormulaError:
        return None


def classify(formula: str, statuses: dict[str, str]) -> str:
    """Three-way applicability decision for one rule under its gate evidence.

    ``not_applicable`` when the formula is already False on presence alone.
    Otherwise the question is whether the *confirmed* atoms carry the rule by
    themselves: re-evaluate with every unresolved positive atom pessimistically
    bound to absent. Still not False means the rule survives regardless of what
    we failed to see, so it is admitted; False means the answer hinges on an atom
    we could not resolve, so the gate abstains rather than guessing.
    """
    if gate_outcome(formula, statuses) is False:
        return NOT_APPLICABLE
    pessimistic = {
        atom: (statuses.get(atom, "need_review")
               if STATUS_TO_VALUE.get(statuses.get(atom, "need_review")) is not None
               else "violated")
        for atom in positive_atoms(formula)
    }
    return UNKNOWN if gate_outcome(formula, pessimistic) is False else APPLICABLE


def gate_payload(rule: dict[str, Any]) -> dict[str, Any]:
    """The prompt payload for one rule: identity plus only its positive checkpoints."""
    formula = rule.get("visual_screening_rule") or ""
    checkpoints = rule.get("visual_checkpoints") or {}
    atoms = positive_atoms(formula)
    return {
        "rule_id": rule["rule_id"],
        "rule_name": rule.get("rule_name"),
        "major_category": rule.get("major_category"),
        "subcategory": rule.get("subcategory"),
        "applicability_checkpoints": {
            atom: checkpoints.get(atom, atom) for atom in atoms
        },
    }


def statuses_from_evidence(evidence: dict[str, Any], atoms: list[str]) -> dict[str, str]:
    """Pull the positive atoms' statuses out of a gate response, defaulting to unknown."""
    reported = {}
    for item in evidence.get("checkpoint_evidence") or []:
        if isinstance(item, dict) and item.get("checkpoint"):
            reported[item["checkpoint"]] = item.get("status", "need_review")
    return {atom: reported.get(atom, "need_review") for atom in atoms}
