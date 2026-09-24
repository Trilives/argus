"""Site-grouped folds over confirmed sites, with near-duplicate clusters kept whole.

A fold boundary must never separate two photographs of one site, nor two
near-duplicates of one scene. Sites are unioned with every near-duplicate cluster
that touches them, so a cluster spanning two sites fuses those sites into one unit.
Images without a confirmed site are excluded, never pooled into a residual site.
Everything here is label-free: folds are balanced on image counts, not outcomes.
"""
from __future__ import annotations

import hashlib

DEFAULT_FOLDS = 5
SEED = 20260922


def site_of(confirmed: dict) -> dict[str, str]:
    """``image_id -> site_key`` from ``site_keys_confirmed.json``; duplicates are errors."""
    out: dict[str, str] = {}
    for site in confirmed["sites"]:
        for image in site["images"]:
            image_id = image["image_id"]
            if image_id in out:
                raise ValueError(f"image assigned to two sites: {image_id}")
            out[image_id] = site["site_key"]
    return out


def units(sites: dict[str, str], clusters: list[list[str]]) -> list[dict]:
    """Union sites through near-duplicate clusters; each unit is indivisible."""
    parent = {key: key for key in set(sites.values())}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    fused = 0
    for cluster in clusters:
        keys = sorted({sites[i] for i in cluster if i in sites})
        for other in keys[1:]:
            a, b = find(keys[0]), find(other)
            if a != b:
                parent[b] = a
                fused += 1
    grouped: dict[str, dict] = {}
    for image_id, key in sorted(sites.items()):
        unit = grouped.setdefault(find(key), {"unit": find(key), "sites": set(), "images": []})
        unit["sites"].add(key)
        unit["images"].append(image_id)
    return [{"unit": u["unit"], "sites": sorted(u["sites"]), "images": u["images"],
             "fused_by_near_duplicate": len(u["sites"]) > 1}
            for u in sorted(grouped.values(), key=lambda u: u["unit"])]


def _tiebreak(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def assign_folds(unit_rows: list[dict], k: int = DEFAULT_FOLDS, seed: int = SEED) -> dict[str, int]:
    """Largest unit first into the lightest fold; hashed-key tie-breaks, no outcomes read."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 2 or k > len(unit_rows):
        raise ValueError("k must be an integer in [2, number of units]")
    load = [0] * k
    fold_of: dict[str, int] = {}
    ordered = sorted(unit_rows, key=lambda u: (-len(u["images"]), _tiebreak(seed, u["unit"])))
    for unit in ordered:
        fold = min(range(k), key=lambda f: (load[f], f))
        load[fold] += len(unit["images"])
        for image_id in unit["images"]:
            fold_of[image_id] = fold
    return fold_of


def partition(confirmed: dict, clusters: list[list[str]], population: set[str],
              k: int = DEFAULT_FOLDS, seed: int = SEED) -> dict:
    """Folds restricted to ``population``, plus the counts a methods section reports."""
    sites = {i: s for i, s in site_of(confirmed).items() if i in population}
    rows = units(sites, clusters)
    folds = assign_folds(rows, k, seed)
    sizes = [sum(f == j for f in folds.values()) for j in range(k)]
    return {"k": k, "seed": seed, "folds": folds, "units": rows,
            "counts": {"population": len(population), "assigned": len(folds),
                       "excluded_no_site": len(population) - len(folds),
                       "sites": len(set(sites.values())), "units": len(rows),
                       "fused_units": sum(u["fused_by_near_duplicate"] for u in rows),
                       "fold_sizes": sizes}}


def split_leaks(folds: dict[str, int], sites: dict[str, str],
                clusters: list[list[str]]) -> list[str]:
    """Every site or cluster that crosses a fold boundary; empty means leak-free."""
    leaks = []
    by_site: dict[str, set[int]] = {}
    for image_id, fold in folds.items():
        by_site.setdefault(sites[image_id], set()).add(fold)
    leaks += [f"site:{s}" for s, fs in sorted(by_site.items()) if len(fs) > 1]
    for cluster in clusters:
        fs = {folds[i] for i in cluster if i in folds}
        if len(fs) > 1:
            leaks.append("cluster:" + ",".join(sorted(cluster)))
    return leaks
