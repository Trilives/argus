"""Atom posteriors: log-prob bucketing, answer-token location, and question-set coverage.

No model is called. The token streams mirror what the local Qwen3.5-9B server emits.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom_posterior import (BANNED, answer_positions, bucket, build_prompt, posteriors,  # noqa: E402
                            state_atoms, validate_state)
from typed_schema import compile_schema  # noqa: E402

CONFIG = "data/rules/proposed/rcasr_v1.json"


def compiled() -> dict:
    return compile_schema(json.loads((ROOT / "data/rules/rules_en.json").read_text()),
                          json.loads((ROOT / "data/rules/proposed/typed_evidence_v1.json").read_text()))


def stream(pairs: list[tuple[str, str, list[tuple[str, float]]]]) -> list:
    """(key, answer, alternatives) -> the token stream a JSON reply produces."""
    tokens = [("{", []), ("\n", [])]
    for key, answer, alternatives in pairs:
        tokens += [(' "', []), (key, []), ('":', []), (' "', []), (answer, alternatives), ('",', []),
                   ("\n", [])]
    return tokens + [("}", [])]


class BucketTests(unittest.TestCase):
    def test_prefix_tokens_are_bucketed_and_renormalised(self) -> None:
        out = bucket([("no", math.log(.6)), ("unc", math.log(.2)), ("yes", math.log(.1)),
                      ("false", math.log(.1))])
        self.assertAlmostEqual(out["p_no"], .6 / .9)
        self.assertAlmostEqual(out["p_unclear"], .2 / .9)
        self.assertAlmostEqual(sum(out.values()), 1.0)

    def test_no_mass_on_any_answer_is_unknown(self) -> None:
        self.assertIsNone(bucket([("false", 0.0), ("maybe", -1.0)]))


class PositionTests(unittest.TestCase):
    def test_answer_token_follows_its_key(self) -> None:
        tokens = stream([("a_present", "yes", [("yes", -0.1)]), ("b_present", "no", [("no", -0.2)])])
        found = answer_positions(tokens, {"a_present", "b_present"})
        self.assertEqual(found["a_present"], [("yes", -0.1)])
        self.assertEqual(found["b_present"], [("no", -0.2)])

    def test_missing_id_stays_unknown(self) -> None:
        tokens = stream([("a_present", "yes", [("yes", 0.0)])])
        out = posteriors(tokens, {"a_present": "q", "b_present": "q"})
        self.assertIsNone(out["b_present"])
        self.assertAlmostEqual(out["a_present"]["p_yes"], 1.0)

    def test_unexpected_ids_are_ignored(self) -> None:
        tokens = stream([("zzz", "yes", [("yes", 0.0)])])
        self.assertEqual(answer_positions(tokens, {"a_present"}), {})


class QuestionSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / CONFIG).read_text())
        cls.compiled = compiled()

    def test_state_questions_cover_every_non_gate_visual_atom(self) -> None:
        validate_state(self.config, self.compiled)
        state = self.config["passes"]["state"]
        self.assertEqual(sorted({*state["questions"], *state["unknown_only"]}), state_atoms(self.compiled))

    def test_prompt_never_names_a_rule_or_a_violation(self) -> None:
        """Question texts and template are clean. Two atom ids the reply must echo contain
        'compliant' as an attribute name; they name no rule or provision. Pinned here so a
        new leaking id fails loudly instead of slipping in (disclosed in the RCASR record)."""
        state = self.config["passes"]["state"]
        prompt = build_prompt(state["questions"], state["instruction_template"]).lower()
        for name in ("clothing_compliant", "guardrail_compliant"):
            prompt = prompt.replace(name, "")
        for word in BANNED:
            self.assertNotIn(word, prompt)

    def test_external_atom_cannot_be_asked(self) -> None:
        config = json.loads(json.dumps(self.config))
        state = config["passes"]["state"]
        state["questions"]["scrap_condition_met"] = "Scrap condition met."
        state["unknown_only"] = []
        with self.assertRaises(ValueError):
            validate_state(config, self.compiled)


if __name__ == "__main__":
    unittest.main()
