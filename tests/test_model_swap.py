"""Model swap helpers: the cross-arm bootstrap keeps each arm's own verdicts, and the
pre-specified C2 wording follows the sign of the CI."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_lowwidth_e2 import paired_e2e_ci  # noqa: E402
from eval_model_swap import cross_arm_ci, wording  # noqa: E402

STATUSES = {"a": {"r1": "non_compliant", "r2": "compliant"},
            "b": {"r1": "non_compliant", "r3": "non_compliant"},
            "c": {}}
UNITS = [["a"], ["b", "c"]]
SELECTED = {"a": ["r1", "r2"], "b": ["r1", "r3"], "c": ["r1"]}
HIT_ALL = {("a", "r1"): "non_compliant", ("a", "r2"): "compliant", ("b", "r1"): "non_compliant",
           ("b", "r3"): "non_compliant", ("c", "r1"): "compliant"}
MISS_ALL = {k: "compliant" for k in HIT_ALL}


def test_identical_arms_give_a_zero_interval():
    ci = cross_arm_ci(UNITS, SELECTED, HIT_ALL, SELECTED, HIT_ALL, STATUSES, seed=1, repeats=200)
    assert ci == {"gv_recall_delta_ci": [0.0, 0.0], "fa_per_image_delta_ci": [0.0, 0.0]}


def test_same_selection_different_verdicts_is_a_recall_difference():
    ci = cross_arm_ci(UNITS, SELECTED, HIT_ALL, SELECTED, MISS_ALL, STATUSES, seed=1, repeats=200)
    assert ci["gv_recall_delta_ci"] == [1.0, 1.0]
    assert ci["fa_per_image_delta_ci"] == [0.0, 0.0]


def test_matches_the_single_verdict_bootstrap_when_verdicts_agree():
    other = {"a": ["r1"], "b": ["r3"], "c": []}
    mine = cross_arm_ci(UNITS, SELECTED, HIT_ALL, other, HIT_ALL, STATUSES, seed=7, repeats=300)
    theirs = paired_e2e_ci(UNITS, SELECTED, other, HIT_ALL, STATUSES, seed=7, repeats=300)
    assert mine["gv_recall_delta_ci"] == theirs["gv_recall_delta_ci"]
    assert mine["fa_per_image_delta_ci"] == theirs["fa_per_image_delta_ci"]


def test_wording_follows_the_interval():
    assert wording([0.01, 0.05], "gain", "cost") == "gain"
    assert wording([-0.05, -0.01], "gain", "cost") == "cost"
    assert wording([-0.02, 0.03], "gain", "cost") == "no detectable difference from the reference"
