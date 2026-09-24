"""Low-width E2 helpers: subset assertion and the paired site-unit end-to-end bootstrap."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_lowwidth_e2 import assert_nested, paired_e2e_ci, unit_rows  # noqa: E402

STATUSES = {"a": {"r1": "non_compliant", "r2": "compliant"},
            "b": {"r1": "non_compliant", "r3": "non_compliant"},
            "c": {}}
VERDICTS = {("a", "r1"): "non_compliant", ("a", "r2"): "non_compliant", ("a", "r3"): "compliant",
            ("b", "r1"): "non_compliant", ("b", "r3"): "compliant", ("b", "r2"): "non_compliant",
            ("c", "r1"): "non_compliant", ("c", "r2"): "compliant"}
UNITS = [["a"], ["b", "c"]]


def test_assert_nested_counts_pairs_and_rejects_escapes():
    inner = {"a": ["r1"], "b": ["r1", "r3"], "c": []}
    outer = {"a": ["r1", "r2"], "b": ["r1", "r3"], "c": ["r1"]}
    assert assert_nested(inner, outer) == 3
    with unittest.TestCase().assertRaises(AssertionError):
        assert_nested({"a": ["r3"], "b": [], "c": []}, outer)


def test_unit_rows_sum_hits_support_and_false_alarms_per_unit():
    left = {"a": ["r1", "r2"], "b": ["r1"], "c": ["r1"]}
    right = {"a": ["r3"], "b": ["r3", "r2"], "c": ["r2"]}
    rows = unit_rows(UNITS, left, right, VERDICTS, STATUSES)
    # unit a: left hits 1 (r1), right hits 0; support 1; left FA 1 (r2 recorded compliant), right FA 0
    assert rows[0].tolist() == [1, 0, 1, 1, 0, 1]
    # unit b+c: left hits 1 (b r1), right 0; support 2; left FA 1 (c r1), right FA 1 (b r2); 2 images
    assert rows[1].tolist() == [1, 0, 2, 1, 1, 2]


def test_paired_ci_brackets_the_point_estimate_and_is_seeded():
    left = {"a": ["r1", "r2"], "b": ["r1"], "c": ["r1"]}
    right = {"a": ["r3"], "b": ["r3", "r2"], "c": ["r2"]}
    ci = paired_e2e_ci(UNITS, left, right, VERDICTS, STATUSES, seed=1, repeats=200)
    lo, hi = ci["gv_recall_delta_ci"]
    assert lo <= (2 - 0) / 3 <= hi
    assert ci == paired_e2e_ci(UNITS, left, right, VERDICTS, STATUSES, seed=1, repeats=200)
    assert ci["n_units"] == 2


class LowWidthE2Tests(unittest.TestCase):
    """unittest entry point for the plain test functions above."""

    def test_assert_nested(self) -> None:
        test_assert_nested_counts_pairs_and_rejects_escapes()

    def test_unit_rows(self) -> None:
        test_unit_rows_sum_hits_support_and_false_alarms_per_unit()

    def test_paired_ci(self) -> None:
        test_paired_ci_brackets_the_point_estimate_and_is_seeded()
