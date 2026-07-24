"""Authoritative image -> rule gold set (recall gold).

Per ``docs/EXPERIMENT_PROTOCOL.md`` §2 the sole authoritative gold source is
``data/annotations/image_rule_gold.json`` (annotated, ``reviewed == true``).
Images are annotated directly there; there is no seed label file to fall back
to, so an absent or unreviewed gold set yields nothing rather than a weaker
substitute silently standing in for it.

The recall gold set is the full ``rule_ids`` list regardless of compliance
status: an ``undetermined`` rule is still a rule the retriever should recall;
its status only matters for the judgement-layer evaluation, which reads
``rule_statuses`` from the same file.
"""

from __future__ import annotations

from pathlib import Path

import paths
from io_utils import load_json

GOLD_PATH = paths.DATA_DIR / "annotations" / "image_rule_gold.json"


def load_gold_rule_ids(gold_path: Path | str = GOLD_PATH) -> dict[str, list[str]]:
    """Return ``{image_id: [rule_id, ...]}`` for the recall gold set.

    Only ``reviewed`` entries with a non-empty rule set count as gold.
    """
    gold_path = Path(gold_path)
    if not gold_path.exists():
        return {}
    doc = load_json(gold_path)
    out: dict[str, list[str]] = {}
    for image_id, entry in doc.get("gold", {}).items():
        if entry.get("reviewed") and entry.get("rule_ids"):
            out[Path(str(image_id)).stem] = list(entry["rule_ids"])
    return out


def load_gold_statuses(gold_path: Path | str = GOLD_PATH) -> dict[str, dict[str, str]]:
    """Return ``{image_id: {rule_id: status}}`` for the judgement gold.

    Empty when no calibrated gold exists (seed labels carry no statuses).
    """
    gold_path = Path(gold_path)
    if not gold_path.exists():
        return {}
    doc = load_json(gold_path)
    out: dict[str, dict[str, str]] = {}
    for image_id, entry in doc.get("gold", {}).items():
        if entry.get("reviewed") and entry.get("rule_statuses"):
            out[Path(str(image_id)).stem] = dict(entry["rule_statuses"])
    return out
