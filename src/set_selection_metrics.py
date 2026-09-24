"""Observed-label summaries and paired duplicate-cluster diagnostic intervals."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def metrics(selected: dict[str, list[str]], positives: dict[str, set[str]],
            family: dict[str, str], cap: int = 3) -> dict:
    if not selected or set(selected) != set(positives):
        raise ValueError("nonempty, identical image populations required")
    ids = sorted(selected)
    widths = np.array([len(set(selected[i])) for i in ids])
    support = np.array([len(positives[i]) for i in ids])
    hits = np.array([len(set(selected[i]) & positives[i]) for i in ids])
    losses = (support - hits) / np.maximum(1, support)
    by_rule = {}
    for rule in sorted(set().union(*positives.values())):
        n = sum(rule in p for p in positives.values())
        hit = sum(rule in positives[i] and rule in selected[i] for i in ids)
        by_rule[rule] = {"positive_support": n, "hits": hit, "recall": hit / n}
    by_family = {}
    for group in sorted(set(family.values())):
        rows = [v for r, v in by_rule.items() if family[r] == group]
        n = sum(v["positive_support"] for v in rows)
        hit = sum(v["hits"] for v in rows)
        by_family[group] = {"positive_support": n, "hits": hit, "recall": hit / n if n else None}
    recalls = [v["recall"] for v in by_family.values() if v["recall"] is not None]
    return {"n_images": len(ids), "n_positive_images": int(sum(support > 0)),
            "n_empty_positive_images": int(sum(support == 0)), "positive_pairs": int(sum(support)),
            "hits": int(sum(hits)), "recall_micro": float(sum(hits) / sum(support)) if sum(support) else None,
            "risk": float(losses.mean()),
            "risk_positive_only": float(losses[support > 0].mean()) if any(support > 0) else None,
            "width_mean": float(widths.mean()), "width_median": float(np.median(widths)),
            "width_p95": float(np.quantile(widths, .95)), "width_total": int(sum(widths)),
            "cap_review_rate": float(np.mean(widths > cap)), "empty_selection_rate": float(np.mean(widths == 0)),
            "macro_rule_recall": float(np.mean([v["recall"] for v in by_rule.values()])) if by_rule else None,
            "macro_family_recall": float(np.mean(recalls)) if recalls else None,
            "by_rule": by_rule, "by_family": by_family}


def cluster_index(ids: list[str], clusters: list[list[str]]) -> dict[str, str]:
    """Union across full-pool clusters before restricting to the scored population."""
    parent = {i: i for i in set(ids).union(*(set(c) for c in clusters))}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for cluster in clusters:
        for member in cluster[1:]:
            a, b = root(cluster[0]), root(member)
            parent[max(a, b)] = min(a, b)
    return {i: root(i) for i in ids}


def paired_intervals(left: dict[str, list[str]], right: dict[str, list[str]],
                     positives: dict[str, set[str]], clusters: list[list[str]],
                     seed: int = 20260919, repeats: int = 2000) -> dict:
    """Conditional diagnostic bootstrap; never labeled site-clustered or confirmatory."""
    if not left or set(left) != set(right) or set(left) != set(positives) or repeats < 1:
        raise ValueError("nonempty aligned populations and positive repeats required")
    groups = defaultdict(list)
    for image_id, group in cluster_index(sorted(left), clusters).items():
        groups[group].append(image_id)
    rows = []
    for group in sorted(groups):
        ids = groups[group]
        rows.append([sum(len(set(left[i]) & positives[i]) - len(set(right[i]) & positives[i]) for i in ids),
                     sum(len(positives[i]) for i in ids),
                     sum((len(positives[i] - set(left[i])) - len(positives[i] - set(right[i]))) /
                         max(1, len(positives[i])) for i in ids), len(ids),
                     sum(len(positives[i] - set(left[i])) / max(1, len(positives[i])) for i in ids)])
    array = np.array(rows, dtype=float)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), (repeats, len(array)))].sum(axis=1)
    valid = draws[:, 1] > 0
    interval = lambda v: [float(x) for x in np.quantile(v, [.025, .975])] if len(v) else None
    return {"unit": "near_duplicate_component_not_site", "n_clusters": len(groups),
            "repeats": repeats, "seed": seed, "recall_delta_ci": interval(draws[valid, 0] / draws[valid, 1]),
            "risk_delta_ci": interval(draws[:, 2] / draws[:, 3]),
            "left_risk_ci": interval(draws[:, 4] / draws[:, 3]),
            "undefined_recall_replicates": int(sum(~valid)),
            "interpretation": "descriptive, conditional on fitted thresholds; not independent validation"}
