"""Selection must retain tied scores, unknown labels and deferred candidates."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from set_selection import calibrate, capped_selection, matched_rank_sets, select


class SelectionTests(unittest.TestCase):
    def test_threshold_includes_ties(self):
        self.assertEqual(select({"b": .5, "a": .5, "c": .1}, .5), ["a", "b"])

    def test_crc_finite_sample_correction(self):
        scores = {str(i): {"positive": .8, "unknown": .2} for i in range(9)}
        gold = {i: {"positive"} for i in scores}
        fit = calibrate(scores, gold, .1, (0., .2, .8, 1.))
        self.assertEqual(fit["threshold"], .8)
        self.assertEqual(fit["corrected_risk"], .1)
        self.assertEqual(fit["empirical_risk"], 0.)
        self.assertIsNone(calibrate(scores, gold, .09)["threshold"])

    def test_empty_positive_images_are_zero_loss_not_negatives(self):
        fit = calibrate({"a": {"r": .5}, "b": {"r": .5}},
                        {"a": {"r"}, "b": set()}, .5, (0., 1.))
        self.assertEqual(fit["threshold"], 0.)
        self.assertEqual(fit["n_empty_positive"], 1)

    def test_invalid_input_fails_closed(self):
        for alpha in (-.1, 1., float("nan")):
            with self.assertRaises(ValueError):
                calibrate({"a": {"r": .5}}, {"a": {"r"}}, alpha)
        for scores, gold in [({}, {}), ({"a": {"r": .5}}, {"b": {"r"}}),
                             ({"a": {"r": float("nan")}}, {"a": {"r"}}),
                             ({"a": {"r": .5}}, {"a": {"unknown_rule"}})]:
            with self.assertRaises(ValueError):
                calibrate(scores, gold, .1)

    def test_cap_defers_without_truncating(self):
        result = capped_selection({"a": .9, "b": .8, "c": .7}, .5, cap=2)
        self.assertEqual(result["status"], "need_review")
        self.assertEqual(result["selected"], ["a", "b", "c"])
        self.assertEqual(result["automatic_candidates"], [])
        self.assertEqual(capped_selection({"r": .8}, None)["status"], "need_review")
        self.assertEqual(capped_selection({"r": .8}, .9)["status"], "empty_selection")

    def test_matched_width_is_exact_and_order_independent(self):
        scores = {i: {"a": .9, "b": .6, "c": .2} for i in ("z", "x", "y")}
        sets = matched_rank_sets(scores, 5, seed=7)
        self.assertEqual(sum(map(len, sets.values())), 5)
        self.assertEqual(sets, matched_rank_sets(dict(reversed(list(scores.items()))), 5, seed=7))
        self.assertEqual(sorted(map(len, sets.values())), [1, 2, 2])
        with self.assertRaises(ValueError):
            matched_rank_sets(scores, 10)


if __name__ == "__main__":
    unittest.main()
