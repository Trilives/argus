"""Unstructured set-selection baselines; no inference, gold backfilling or risk claim.

CRC controls a bounded monotone observed-positive loss under exchangeability and
proper data separation. This module does not establish those assumptions for callers.
"""
from __future__ import annotations

import hashlib
import math

THRESHOLD_GRID = tuple(j / 500 for j in range(501))
ALPHAS = (.01, .025, .05, .10, .15, .20, .30, .50)
FIXED_K = (1, 2, 3, 5, 10, 20, 42)
FIXED_THRESHOLDS = (.50, .75, .90, .95, .975)
SEED = 20260919


def validate_scores(scores: dict[str, float]) -> None:
    if not scores or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores.values()):
        raise ValueError("scores must be a nonempty full-library map of finite values in [0, 1]")


def select(scores: dict[str, float], threshold: float) -> list[str]:
    """Inclusive score threshold, deterministically ordered, no hidden top-k cap."""
    validate_scores(scores)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite in [0, 1]")
    return sorted((r for r, s in scores.items() if s >= threshold),
                  key=lambda r: (-scores[r], r))


def calibrate(scores: dict[str, dict[str, float]], positives: dict[str, set[str]],
              alpha: float, grid: tuple[float, ...] = THRESHOLD_GRID) -> dict:
    """Largest threshold with (sum of per-image losses + 1)/(n+1) <= alpha.

    The threshold grid must be fixed independently of calibration observations.
    No admissible threshold returns None, which inference must defer for review.
    The returned correction is a calibration statistic, not a test-cohort CI.
    """
    if not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("alpha must be finite in [0, 1)")
    if not scores or set(scores) != set(positives):
        raise ValueError("nonempty, identical image populations required")
    if not grid or tuple(sorted(set(grid))) != grid or grid[0] < 0 or grid[-1] > 1:
        raise ValueError("grid must be nonempty, ascending and unique in [0, 1]")
    if any(not math.isfinite(t) for t in grid):
        raise ValueError("grid must be finite")
    library = set(next(iter(scores.values())))
    positive_scores = []
    for image_id, row in scores.items():
        validate_scores(row)
        if set(row) != library or not positives[image_id] <= library:
            raise ValueError("all scores and positive labels must use the same complete library")
        positive_scores.append([row[r] for r in positives[image_id]])
    n = len(scores)
    out = {"alpha": alpha, "n": n, "n_empty_positive": sum(not p for p in positive_scores),
           "threshold": None, "corrected_risk": None, "empirical_risk": None}
    for threshold in grid:
        total = math.fsum(sum(s < threshold for s in row) / max(1, len(row))
                          for row in positive_scores)
        correction = (total + 1) / (n + 1)
        if correction <= alpha:
            out.update(threshold=threshold, corrected_risk=correction, empirical_risk=total / n)
    return out


def capped_selection(scores: dict[str, float], threshold: float | None,
                     cap: int = 3) -> dict:
    """Retain the full set for audit; defer the entire image when over budget."""
    validate_scores(scores)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError("cap must be a nonnegative integer")
    if threshold is None:
        return {"status": "need_review", "reason": "calibration_infeasible",
                "selected": [], "automatic_candidates": []}
    chosen = select(scores, threshold)
    deferred = len(chosen) > cap
    return {"status": "need_review" if deferred else "selected" if chosen else "empty_selection",
            "reason": "candidate_cap" if deferred else None, "selected": chosen,
            "automatic_candidates": [] if deferred else chosen}


def matched_rank_sets(scores: dict[str, dict[str, float]], total: int,
                      seed: int = SEED) -> dict[str, list[str]]:
    """Label-free floor/ceiling-k mixture at an exact total candidate budget.

    The allocation order is hashed image ID, not image difficulty. This comparator
    needs no gold. It is a rank mixture, not a pure fixed-k or adaptive score policy.
    """
    if not scores or isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("nonempty population and nonnegative integer budget required")
    library = set(next(iter(scores.values())))
    for row in scores.values():
        validate_scores(row)
        if set(row) != library:
            raise ValueError("matched rank mixture requires full-library scores")
    if total > len(scores) * len(library):
        raise ValueError("budget exceeds library capacity")
    floor, remainder = divmod(total, len(scores))
    order = sorted(scores, key=lambda i: (hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(), i))
    extra = set(order[:remainder])
    return {i: select(row, 0.)[:floor + (i in extra)] for i, row in scores.items()}
