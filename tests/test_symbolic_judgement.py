"""J3-sym: three-valued formula evaluation + symbolic verdicts (P1 item)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

CS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CS_ROOT / "src"))

import symbolic_judgement as sj  # noqa: E402
from io_utils import load_json  # noqa: E402
import paths  # noqa: E402

FORMULA = (
    "opening_present == yes AND ((cover_present == no OR cover_fixed == no) "
    "OR protection_intact == no)"
)


def _ev(statuses: dict[str, str], rule_id: str = "R-T-001") -> dict:
    return {
        "rule_id": rule_id,
        "rule_name": "test rule",
        "visual_screening_rule": FORMULA,
        "checkpoint_evidence": [
            {"checkpoint": k, "status": v, "confidence": 0.9} for k, v in statuses.items()
        ],
        "missing_information": [],
    }


class EvalFormulaTest(unittest.TestCase):
    def test_violation_confirmed(self):
        v = sj.eval_formula(FORMULA, {"opening_present": "satisfied", "cover_present": "violated",
                                      "cover_fixed": "satisfied", "protection_intact": "satisfied"})
        self.assertIs(v, True)

    def test_compliant_confirmed(self):
        v = sj.eval_formula(FORMULA, {"opening_present": "satisfied", "cover_present": "satisfied",
                                      "cover_fixed": "satisfied", "protection_intact": "satisfied"})
        self.assertIs(v, False)

    def test_subject_absent_makes_formula_false(self):
        # gate == yes with status violated (subject absent) -> false regardless of unknowns
        v = sj.eval_formula(FORMULA, {"opening_present": "violated", "cover_present": "not_visible"})
        self.assertIs(v, False)

    def test_unknown_propagates_kleene(self):
        # gate satisfied, defect side unknown -> unknown
        v = sj.eval_formula(FORMULA, {"opening_present": "satisfied", "cover_present": "not_visible",
                                      "cover_fixed": "satisfied", "protection_intact": "satisfied"})
        self.assertIsNone(v)

    def test_kleene_or_true_dominates_unknown(self):
        v = sj.eval_formula(FORMULA, {"opening_present": "satisfied", "cover_present": "not_visible",
                                      "cover_fixed": "violated"})
        self.assertIs(v, True)

    def test_missing_checkpoint_is_unknown(self):
        self.assertIsNone(sj.eval_formula(FORMULA, {"opening_present": "satisfied"}))

    def test_malformed_formula_raises(self):
        with self.assertRaises(sj.FormulaError):
            sj.eval_formula("opening_present == maybe", {})
        with self.assertRaises(sj.FormulaError):
            sj.eval_formula("(a == yes", {"a": "satisfied"})


class SymbolicVerdictTest(unittest.TestCase):
    def test_violation_verdict(self):
        out = sj.symbolic_verdict(_ev({"opening_present": "satisfied", "cover_present": "violated",
                                       "cover_fixed": "violated", "protection_intact": "violated"}))
        self.assertEqual(out["compliance_judgement"]["compliance_label"], "non_compliant")
        self.assertEqual(out["matched_rule"]["rule_id"], "R-T-001")

    def test_absent_subject_defaults_compliant(self):
        # gate not visible -> no_subject_default, the family-arm rejection path
        out = sj.symbolic_verdict(_ev({"opening_present": "not_visible", "cover_present": "violated",
                                       "cover_fixed": "violated", "protection_intact": "violated"}))
        self.assertEqual(out["compliance_judgement"]["compliance_label"], "compliant")
        self.assertEqual(out["symbolic"]["route"], "no_subject_default")

    def test_non_gate_unknown_routes_need_review(self):
        out = sj.symbolic_verdict(_ev({"opening_present": "satisfied", "cover_present": "need_review",
                                       "cover_fixed": "satisfied", "protection_intact": "satisfied"}))
        self.assertEqual(out["compliance_judgement"]["compliance_label"], "need_review")
        self.assertIn("cover_present", out["symbolic"]["unknown_atoms"])

    def test_not_evaluable_sentinel_abstains(self):
        item = _ev({})
        item["visual_screening_rule"] = sj.NOT_EVALUABLE
        out = sj.symbolic_verdict(item)
        self.assertEqual(out["compliance_judgement"]["compliance_label"], "need_review")
        self.assertEqual(out["symbolic"]["route"], "not_visually_evaluable")

    def test_strict_prompt_variant_exists_and_differs(self):
        import prompts

        base = prompts.rule_evidence_prompt()
        strict = prompts.rule_evidence_strict_prompt()
        self.assertNotEqual(base, strict)
        self.assertIn("exactly this type", strict)
        # only the subject-gate item differs; the output schema stays identical
        self.assertEqual(base.split("Output exactly one JSON")[1],
                         strict.split("Output exactly one JSON")[1])

    def test_all_rule_formulas_parse(self):
        for rule in load_json(paths.RULES_PATH):
            formula = rule["visual_screening_rule"]
            if formula.strip() == sj.NOT_EVALUABLE:
                continue
            tree = sj.parse_formula(formula)
            self.assertIsNotNone(tree, rule["rule_id"])
            atoms = sj.formula_atoms(formula)
            self.assertTrue(set(atoms) <= set(rule["visual_checkpoints"]), rule["rule_id"])


if __name__ == "__main__":
    unittest.main()
