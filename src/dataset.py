"""Dataset loading for CS evidence-chain experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import paths
from io_utils import load_json


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: Path
    label: str
    source_folder: str
    original_name: str
    split: str

    def to_task_meta(self) -> dict[str, str]:
        return {
            "image_id": self.image_id,
            "image_path": str(self.image_path),
            "label": self.label,
            "source_folder": self.source_folder,
            "original_name": self.original_name,
            "split": self.split,
        }


def load_image_mapping() -> dict[str, dict[str, str]]:
    mapping_path = paths.IMAGE_MAPPING_PATH
    rows: dict[str, dict[str, str]] = {}
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["anonymous_name"]] = {
                "label": row.get("label", ""),
                "source_folder": row.get("source_folder", ""),
                "original_name": row.get("original_name", ""),
            }
    return rows


def load_split_file(split_name: str) -> list[str]:
    split_path = paths.SPLITS_DIR / f"{split_name}_images.txt"
    if not split_path.exists():
        split_json = load_json(paths.SPLITS_DIR / "image_split.json")
        splits = split_json.get("splits", {}) if isinstance(split_json, dict) else {}
        if split_name not in splits:
            raise ValueError(f"Unknown split {split_name!r}")
        return list(splits[split_name])
    return [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_split_records(split_name: str, *, limit: int = 0) -> list[ImageRecord]:
    mapping = load_image_mapping()
    image_names = load_split_file(split_name)
    if limit > 0:
        image_names = image_names[:limit]

    records: list[ImageRecord] = []
    for image_name in image_names:
        meta = mapping.get(image_name, {})
        image_path = paths.IMAGES_DIR / image_name
        records.append(
            ImageRecord(
                image_id=Path(image_name).stem,
                image_path=image_path,
                label=meta.get("label", ""),
                source_folder=meta.get("source_folder", ""),
                original_name=meta.get("original_name", ""),
                split=split_name,
            )
        )
    return records


def dataset_summary(split_name: str, *, limit: int = 0) -> dict[str, Any]:
    records = load_split_records(split_name, limit=limit)
    labels: dict[str, int] = {}
    for record in records:
        labels[record.label] = labels.get(record.label, 0) + 1
    return {
        "split": split_name,
        "limit": limit,
        "image_count": len(records),
        "labels": labels,
    }
