"""Shard assignment for the split RCASR judge: disjoint, complete, order-free."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "retrieval"))

from judge_rcasr_shard import shard_of  # noqa: E402

PAIRS = [(f"img_{i:04d}", f"R-OPN-{j:03d}") for i in range(40) for j in range(6)]


def test_every_pair_lands_in_exactly_one_shard():
    shards = [{p for p in PAIRS if shard_of(p, 2) == k} for k in range(2)]
    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == set(PAIRS)


def test_assignment_does_not_depend_on_order_or_restart():
    first = {p: shard_of(p, 3) for p in PAIRS}
    again = {p: shard_of(p, 3) for p in reversed(PAIRS)}
    assert first == again


def test_both_shards_get_work():
    counts = [sum(shard_of(p, 2) == k for p in PAIRS) for k in range(2)]
    assert min(counts) > len(PAIRS) // 4
