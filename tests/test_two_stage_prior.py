"""Two-stage prior (Todo 3-B): the cascade only reorders stage-1 candidates, and the rank map
is the manuscript's R4 map written for a list of length M."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rcasr import r4_prior  # noqa: E402
from two_stage_prior import cascade, pin, rank_prior  # noqa: E402

LIBRARY = [f"R-{i:02d}" for i in range(12)]


def test_rerank_reorders_only_the_stage1_candidates():
    stage1 = ["R-03", "R-01", "R-07"]
    scores = {rule: -float(i) for i, rule in enumerate(LIBRARY)}  # R-00 would win if it could
    assert cascade(stage1, scores) == ["R-01", "R-03", "R-07"]


def test_rerank_ties_keep_stage1_order():
    assert cascade(["R-05", "R-02", "R-09"], {"R-05": 1.0, "R-02": 2.0, "R-09": 1.0}) == ["R-02", "R-05", "R-09"]


def test_missing_rerank_score_fails_loudly():
    with pytest.raises(KeyError):
        cascade(["R-05", "R-02"], {"R-05": 1.0})


def test_length_four_reproduces_the_frozen_r4_map():
    ranked = ["R-04", "R-00", "R-11", "R-06"]
    assert rank_prior(ranked, LIBRARY, 4) == r4_prior(ranked, LIBRARY)


def test_length_ten_map_is_strictly_decreasing_with_a_lower_floor():
    ranked = LIBRARY[:10]
    prior = rank_prior(ranked, LIBRARY, 10)
    values = [prior[r] for r in ranked]
    assert values == sorted(values, reverse=True) and len(set(values)) == 10
    assert prior["R-10"] == prior["R-11"] == pytest.approx(0.5 / 11)
    assert prior["R-10"] < values[-1]
    assert all(0 < v < 1 for v in prior.values())


def test_list_longer_than_its_declared_length_is_rejected():
    with pytest.raises(ValueError):
        rank_prior(LIBRARY[:5], LIBRARY, 4)


def test_pin_hashes_the_snapshot_and_checks_the_revision(tmp_path):
    repo = tmp_path / "models--org--name"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123")
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"weights")
    hashes = pin("org/name", "abc123", cache=tmp_path)
    assert list(hashes) == ["model.safetensors"] and len(hashes["model.safetensors"]) == 64
    with pytest.raises(ValueError):
        pin("org/name", "other", cache=tmp_path)
