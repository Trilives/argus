"""RCASR scoring invariants: unknown never lowers a score, fitting reads positives only,
and cross-fitting never lets a held-out fold into its own fit or calibration."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcasr import (LOG_FLOOR, features, fit, hard_probability, known, probability,  # noqa: E402
                   r4_prior, score_rows, select_at, width_threshold)
from rcasr_experiment import fold_roles, shuffled, tau_pairs  # noqa: E402

GATE = ("atom", "a", "==", True)
VIS = ("and", (("atom", "a", "==", True), ("atom", "b", "==", False)))
COMPILED = {"rules": {"R1": {"gate": GATE, "visual": VIS}, "R2": {"gate": GATE, "visual": None}}}


def post(p_yes: float, p_no: float, p_unc: float = 0.0) -> dict:
    return {"p_yes": p_yes, "p_no": p_no, "p_unclear": p_unc}


class ProbabilityTests(unittest.TestCase):
    def test_unknown_literal_is_one(self) -> None:
        self.assertEqual(probability(GATE, {}), 1.0)
        self.assertEqual(probability(VIS, {"a": .8}), .8)

    def test_and_or_semantics(self) -> None:
        table = {"a": .5, "b": .2}
        self.assertAlmostEqual(probability(VIS, table), .5 * .8)
        either = ("or", (("atom", "a", "==", True), ("atom", "b", "==", True)))
        self.assertAlmostEqual(probability(either, table), 1 - .5 * .8)

    def test_numeric_comparisons_never_lower_a_score(self) -> None:
        self.assertEqual(probability(("atom", "h_m", ">=", 2), {"h_m": 0.0}), 1.0)

    def test_ternary_lowers_only_on_false(self) -> None:
        self.assertEqual(hard_probability(GATE, {}), 1.0)
        self.assertEqual(hard_probability(GATE, {"a": .1}), LOG_FLOOR)
        self.assertEqual(hard_probability(GATE, {"a": .9}), 1.0)


class AbstentionTests(unittest.TestCase):
    def test_known_requires_confidence_after_unclear_mass(self) -> None:
        self.assertTrue(known(post(.05, .95), .9))
        self.assertFalse(known(post(.05, .55, .4), .9))
        self.assertFalse(known(None, 0))

    def test_unknown_atom_cannot_lower_features(self) -> None:
        low = features(COMPILED, {"a": post(.01, .99, .0)}, .999, .999, set())
        high = features(COMPILED, {"a": post(.01, .99, .0)}, .0, .0, set())
        self.assertEqual(low["R1"], (0.0, 0.0))
        self.assertLess(high["R1"][0], 0.0)

    def test_opening_class_uses_its_own_tau(self) -> None:
        posts = {"a": post(.1, .9)}
        self.assertEqual(features(COMPILED, posts, 0, .99, {"a"})["R1"][0], 0.0)
        self.assertLess(features(COMPILED, posts, .99, 0, {"a"})["R1"][0], 0.0)


class SelectionTests(unittest.TestCase):
    def test_width_threshold_hits_the_budget(self) -> None:
        scores = {"i1": {"R1": .9, "R2": .1}, "i2": {"R1": .8, "R2": .7}}
        chosen = select_at(scores, width_threshold(scores, 1.5))
        self.assertEqual(sum(map(len, chosen.values())), 3)

    def test_tied_block_cannot_break_the_width_budget(self) -> None:
        """Regression (2026-09-22): a tied block used to be admitted whole."""
        library = [f"R{j}" for j in range(40)]
        priors = {f"i{n}": r4_prior(["R0", "R1"], library) for n in range(10)}
        feats = {i: {r: (0.0, 0.0) for r in library} for i in priors}
        scores = score_rows(priors, feats, (1, 1))
        chosen = select_at(scores, width_threshold(scores, 2.5))
        self.assertEqual(sum(map(len, chosen.values())), 25)
        grid = {"theta_gate": [0], "theta_state": [0], "tau": [0], "tau_open": [0]}
        many = {i: {f"R{j}" for j in range(2, 40)} for i in priors}
        best = fit(priors, {(0, 0): feats}, many, 2.5, grid)
        self.assertLessEqual(best["recall"], 5 / 380 + 1e-9)

    def test_r4_prior_is_rank_ordered_and_full_library(self) -> None:
        prior = r4_prior(["R2"], ["R1", "R2"])
        self.assertEqual(prior, {"R1": .1, "R2": .8})

    def test_fit_prefers_evidence_that_separates_positives(self) -> None:
        priors = {"i1": {"R1": .6, "R2": .5}, "i2": {"R1": .6, "R2": .5}}
        feats = {"i1": {"R1": (-5.0, 0.0), "R2": (0.0, 0.0)}, "i2": {"R1": (-5.0, 0.0), "R2": (0.0, 0.0)}}
        grid = {"theta_gate": [0, 1], "theta_state": [0], "tau": [0], "tau_open": [0]}
        best = fit(priors, {(0, 0): feats}, {"i1": {"R2"}, "i2": {"R2"}}, 1.0, grid)
        self.assertEqual(best["theta"], (1, 0))
        self.assertEqual(best["recall"], 1.0)

    def test_scores_stay_in_unit_interval(self) -> None:
        feats = {"i": {"R1": (-9.2, -9.2)}}
        s = score_rows({"i": {"R1": 1e-6}}, feats, (4, 4))["i"]["R1"]
        self.assertTrue(0 <= s <= 1)


class CrossFitTests(unittest.TestCase):
    def test_held_out_and_calibration_folds_never_enter_the_fit(self) -> None:
        folds = {f"i{j}": j % 5 for j in range(20)}
        for held in range(5):
            fitting, calibration, test = fold_roles(folds, 5, held)
            self.assertFalse(set(fitting) & (set(calibration) | set(test)))
            self.assertEqual({folds[i] for i in test}, {held})
            self.assertEqual({folds[i] for i in calibration}, {(held + 1) % 5})

    def test_tau_pairs_override_the_product(self) -> None:
        self.assertEqual(tau_pairs({"tau": [0, .5], "tau_open": [.9]}), [(0, .9), (.5, .9)])
        self.assertEqual(tau_pairs({"tau": [0], "tau_open": [0], "tau_pairs": [[.5, .5]]}), [(.5, .5)])

    def test_null_permutation_keeps_the_multiset(self) -> None:
        posts = {f"i{j}": {"a": j} for j in range(10)}
        out = shuffled(posts, 1)
        self.assertEqual(sorted(v["a"] for v in out.values()), list(range(10)))
        self.assertNotEqual(out, posts)


if __name__ == "__main__":
    unittest.main()
