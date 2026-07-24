"""Visually-confusable rule families and candidate-set expansion.

Some rules share one visual subject and differ only in subtype or orientation
(horizontal vs vertical opening, edge vs guardrail construction, ...); the
visual retriever reliably lands in the right *family* but picks the wrong
member. ``data/rule_assets/rule_families.json`` declares those families (mined
from the R4 confusion analysis); ``expand_candidates`` widens a retrieved
candidate list with the family siblings of its members so the judgement stage
— not the retriever — decides between confusable siblings.
"""
from __future__ import annotations

from pathlib import Path

import paths
from io_utils import load_json

RULE_FAMILIES_PATH = paths.RULE_ASSETS_DIR / "rule_families.json"


def load_family_config(name: str | None = None, path: Path = RULE_FAMILIES_PATH) -> dict[str, list[str]]:
    """One named family config (``None`` -> the file's canonical config)."""
    doc = load_json(path)
    configs = doc["configs"]
    key = name or doc["canonical"]
    if key not in configs:
        raise KeyError(f"unknown family config {key!r}; have {sorted(configs)}")
    return configs[key]


def member_index(families: dict[str, list[str]]) -> dict[str, str]:
    """{rule_id: family_name}; raises if a rule appears in two families."""
    index: dict[str, str] = {}
    for family, members in families.items():
        for rule_id in members:
            if rule_id in index:
                raise ValueError(f"{rule_id} is in both {index[rule_id]!r} and {family!r}")
            index[rule_id] = family
    return index


def expand_candidates(candidates: list[str], families: dict[str, list[str]]) -> list[str]:
    """Append the family siblings of every candidate, order-preserving, deduped.

    The original ranked candidates keep their positions; siblings are appended
    after them (in family-declaration order), so rank-based metrics on the
    unexpanded prefix are unchanged and the judgement stage sees the union.
    """
    index = member_index(families)
    expanded = list(dict.fromkeys(candidates))
    for rule_id in list(expanded):
        family = index.get(rule_id)
        if family:
            expanded += [m for m in families[family] if m not in expanded]
    return expanded
