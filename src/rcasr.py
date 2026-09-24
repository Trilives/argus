"""RCASR scoring: probabilistic gate/state evidence over a retriever prior.

Atom posteriors become literal probabilities, compiled formulas are evaluated with
AND → product and OR → noisy-or, and an unknown atom's literal is 1 so that it can
never lower a score. The score reorders the whole library; a single global threshold
at a width budget turns it into variable-cardinality sets. No label is read here
except by ``fit``, which sees only fitting-fold recorded violations.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np

from typed_predicates import evaluate

LOG_FLOOR = 1e-4
TIE_SEED = 20260922
TIE_EPS = 1e-6  # logit-scale; far below any genuine score difference


def tie_break(image_id: str, rule_id: str) -> float:
    """Label-free hashed offset in [0, TIE_EPS): a strict total order over pairs.

    Without it a block of tied pairs (e.g. R4's unretrieved rules, all at one prior and
    unit evidence) is admitted whole by ``score >= cut`` and the width budget is lost.
    """
    digest = hashlib.sha256(f"{TIE_SEED}:{image_id}:{rule_id}".encode()).digest()
    return TIE_EPS * int.from_bytes(digest[:8], "big") / 2 ** 64


def known(post: dict | None, tau: float) -> bool:
    if post is None:
        return False
    yes, no = post["p_yes"], post["p_no"]
    if yes + no <= 0:
        return False
    q = yes / (yes + no)
    return (1 - post["p_unclear"]) * max(q, 1 - q) >= tau


def literal_table(posts: dict, tau: float, tau_open: float, opening: set[str]) -> dict:
    """``atom -> P(yes)`` for known atoms; unknown atoms are absent."""
    out = {}
    for name, post in posts.items():
        if known(post, tau_open if name in opening else tau):
            out[name] = post["p_yes"] / (post["p_yes"] + post["p_no"])
    return out


def probability(tree: tuple | None, table: dict) -> float:
    """Upper bound over unknown atoms of P(formula), under atom independence."""
    if tree is None:
        return 1.0
    if tree[0] == "atom":
        _, name, op, expected = tree
        if op != "==" or name not in table:
            return 1.0
        return table[name] if expected else 1 - table[name]
    parts = [probability(child, table) for child in tree[1]]
    if tree[0] == "and":
        return math.prod(parts)
    return 1 - math.prod(1 - p for p in parts)


def hard_probability(tree: tuple | None, table: dict) -> float:
    """Ternary ablation: argmax values in Kleene logic; only False lowers the score."""
    if tree is None:
        return 1.0
    values = {name: q >= .5 for name, q in table.items()}
    return LOG_FLOOR if evaluate(tree, values) is False else 1.0


def features(compiled: dict, posts: dict, tau: float, tau_open: float,
             opening: set[str], ternary: bool = False) -> dict[str, tuple[float, float]]:
    """``rule -> (log P_gate, log P_vis)``, clipped at ``log LOG_FLOOR``."""
    table = literal_table(posts, tau, tau_open, opening)
    judge = hard_probability if ternary else probability
    floor = math.log(LOG_FLOOR)
    return {rule: (max(floor, math.log(max(judge(spec["gate"], table), LOG_FLOOR))),
                   max(floor, math.log(max(judge(spec["visual"], table), LOG_FLOOR))))
            for rule, spec in sorted(compiled["rules"].items())}


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def r4_prior(ranked: list[str], library: list[str]) -> dict[str, float]:
    """Label-free map of a ranked R4 list onto the full library (frozen in rcasr_v1)."""
    rank = {rule: i + 1 for i, rule in enumerate(ranked[:4])}
    return {rule: (5 - rank[rule]) / 5 if rule in rank else .1 for rule in library}


def score_rows(priors: dict, feats: dict, theta: tuple[float, float]) -> dict:
    """``image -> rule -> s in (0, 1)``."""
    out = {}
    for image_id, prior in priors.items():
        row = {}
        for rule, p in prior.items():
            gate, vis = feats[image_id][rule]
            z = logit(p) + theta[0] * gate + theta[1] * vis + tie_break(image_id, rule)
            row[rule] = 1 / (1 + math.exp(-z)) if z > -700 else 0.0
        out[image_id] = row
    return out


def width_threshold(scores: dict, width: float) -> float:
    """The score of the (n·width)-th best pair: a global threshold at a mean width."""
    values = np.sort(np.fromiter((s for row in scores.values() for s in row.values()), float))[::-1]
    count = int(round(len(scores) * width))
    if count <= 0:
        return math.inf
    return float(values[min(count, len(values)) - 1])


def select_at(scores: dict, threshold: float) -> dict[str, list[str]]:
    return {i: sorted((r for r, s in row.items() if s >= threshold), key=lambda r: (-row[r], r))
            for i, row in scores.items()}


def recall(selected: dict, positives: dict) -> float:
    hits = sum(len(set(selected[i]) & positives[i]) for i in selected)
    support = sum(len(positives[i]) for i in selected)
    return hits / support if support else 0.0


def _matrix(rows: dict, ids: list[str], library: list[str], pick) -> np.ndarray:
    return np.array([[pick(rows[i][r]) for r in library] for i in ids], dtype=float)


def fit(priors: dict, feature_grid: dict, positives: dict, width: float, grid: dict) -> dict:
    """Recall-maximising (θ_gate, θ_state, τ, τ_open) at a width budget; first grid point wins ties.

    ``feature_grid`` maps ``(τ, τ_open) -> image -> rule -> (gate, vis)`` for the fitting
    images only; ``grid["tau_pairs"]``, when given, replaces the τ × τ_open product. Unlabelled pairs are never negatives: the objective reads recorded
    violations alone. Scores are compared on the logit scale, which orders pairs exactly
    as the sigmoid does.
    """
    ids = sorted(priors)
    library = sorted(next(iter(priors.values())))
    base = _matrix(priors, ids, library, logit) + np.array(
        [[tie_break(i, r) for r in library] for i in ids])
    hit = np.array([[r in positives[i] for r in library] for i in ids])
    support = max(1, int(hit.sum()))
    count = max(1, min(base.size, int(round(len(ids) * width))))
    best = None
    pairs = grid.get("tau_pairs") or list(itertools.product(grid["tau"], grid["tau_open"]))
    for tau, tau_open in pairs:
        gate = _matrix(feature_grid[(tau, tau_open)], ids, library, lambda f: f[0])
        vis = _matrix(feature_grid[(tau, tau_open)], ids, library, lambda f: f[1])
        for tg, ts in itertools.product(grid["theta_gate"], grid["theta_state"]):
            z = base + tg * gate + ts * vis
            cut = np.partition(z.ravel(), -count)[-count]
            value = float(hit[z >= cut].sum()) / support
            key = (tg, ts, tau, tau_open)
            if best is None or value > best["recall"] + 1e-12 or (
                    abs(value - best["recall"]) <= 1e-12 and key < best["key"]):
                best = {"key": key, "theta": (tg, ts), "tau": tau, "tau_open": tau_open,
                        "recall": value}
    return best
