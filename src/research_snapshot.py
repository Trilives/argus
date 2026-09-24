"""Read-only paper-v3 evidence inventory; emits hashes/counts, never private text."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(root: Path, files: list[Path]) -> dict:
    """Whitelist-only hashing. Never serialize environment variables or endpoints."""
    versions = {}
    for name in ("numpy", "matplotlib", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"code_base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "file_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted(set(files))},
            "packages": versions,
            "code_identity": "base commit plus exact working-file hashes; not a clean-commit claim"}


def verify_snapshot(root: Path, frozen: dict) -> None:
    for name, expected in frozen["file_sha256"].items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen input changed or missing: {name}")


def internal_inventory(root: Path) -> dict:
    doc = read_json(root / "data/annotations/image_rule_gold.json")
    rules = read_json(root / "data/rules/rules_en.json")
    library = {r["rule_id"] for r in rules}
    ids = {Path(i).stem for i in doc["images"]}
    if len(ids) != len(doc["images"]) or ids != set(doc["gold"]):
        raise ValueError("gold image population mismatch or duplicate IDs")
    statuses = Counter()
    strata = Counter()
    for item in doc["gold"].values():
        pairs = item["rule_statuses"]
        if not item["reviewed"] or set(pairs) != set(item["rule_ids"]) or not set(pairs) <= library:
            raise ValueError("unreviewed, unsupported or incomplete gold record")
        if set(pairs.values()) - {"compliant", "non_compliant", "undetermined"}:
            raise ValueError("unknown status in frozen gold")
        statuses.update(pairs.values())
        strata["has_recorded_violation" if "non_compliant" in pairs.values() else "no_recorded_violation"] += 1
        for status in ("compliant", "undetermined"):
            strata[f"has_{status}"] += status in pairs.values()
    freeze = read_json(root / "data/annotations/image_rule_gold.manifest.json")
    verify_snapshot(root, freeze)
    image_hashes = {i: sha256(root / "data/images" / f"{i}.jpg") for i in sorted(ids)}
    sites = read_json(root / "data/splits/site_keys_candidate.json")
    date_names = sum(bool(re.search(r"20\d{6}", im["original_name"]))
                     for site in sites["sites"] for im in site["images"])
    return {"n_images": len(ids), "n_rules": len(library), "status_counts": dict(statuses),
            "n_recorded_pairs": sum(statuses.values()), "n_unrecorded_pairs": len(ids) * len(library) - sum(statuses.values()),
            "image_strata_nonexclusive": dict(strata), "historical_freeze_all_hashes_match": True,
            "image_sha256": image_hashes, "development_exposure": "all images",
            "candidate_sites": sites["n_sites"], "site_assigned": sites["n_images_assigned"],
            "site_unassigned": sites["n_gold_images_unassigned"], "site_confirmation": "not established",
            "candidate_filenames_with_date_pattern": date_names, "verified_capture_sessions": None,
            "rule_scopes": dict(Counter(r["decision_scope"] for r in rules)),
            "source_quote_present": sum(bool(r.get("source_quote")) for r in rules),
            "independent_construct_validation": "not established"}


def reliability_inventory(root: Path) -> dict:
    report = read_json(root / "results/annotation/agreement_anno2.json")
    arbitration = read_json(root / "data/annotations/arbitration_decisions_random500.json")
    primary = read_json(root / "data/annotations/image_rule_gold.json")["gold"]
    second = read_json(root / "data/annotations/image_rule_gold_anno2.json")["gold"]
    paired = [(primary[i]["rule_statuses"][r], row["rule_statuses"][r])
              for i, row in second.items() if i in primary
              for r in set(row["rule_statuses"]) & set(primary[i]["rule_statuses"])]
    archived = {k: v for k, v in report.items() if k not in {"disagreements", "annotator", "subset"}}
    return {"historical_pre_adjudication_report": archived,
            "historical_report_hash": sha256(root / "results/annotation/agreement_anno2.json"),
            "raw_pre_adjudication_primary_snapshot": "not located; independent kappa not recomputed",
            "adjudication_applied": arbitration["applied"], "adjudicated_pairs": len(arbitration["decisions"]),
            "post_adjudication_shared_pairs": len(paired),
            "post_adjudication_raw_agreement": sum(a == b for a, b in paired) / len(paired) if paired else None,
            "post_adjudication_primary_status_prevalence": dict(Counter(a for a, _ in paired)),
            "post_adjudication_secondary_status_prevalence": dict(Counter(b for _, b in paired)),
            "qualification": "current primary includes arbitration; current agreement is not independent reliability"}


def native_public_labels(rows: list[dict]) -> list[dict]:
    """Retain native positive IDs; null/empty fields stay unrecorded, not negative."""
    out = []
    seen = set()
    for row in rows:
        image_id = row["image_id"]
        if image_id in seen:
            raise ValueError(f"duplicate public ID: {image_id}")
        seen.add(image_id)
        positive = []
        for number in range(1, 5):
            value = row[f"rule_{number}_violation"]
            if value is not None and not isinstance(value, dict):
                raise ValueError("unexpected public label schema")
            if value:
                positive.append(f"rule_{number}")
        out.append({"image_id": image_id, "recorded_positive_rule_ids": positive,
                    "any_recorded_violation": bool(positive), "unlisted_rule_status": "unknown"})
    return out


def public_inventory(root: Path) -> dict:
    import pyarrow.parquet as pq

    source = root / "data/ConstructionSite-10k/test.parquet"
    columns = ["image_id", *(f"rule_{n}_violation" for n in range(1, 5))]
    derived = native_public_labels(pq.read_table(source, columns=columns).to_pylist())
    ids = {r["image_id"] for r in derived}
    exposed = set()
    files = []
    for path in sorted((root / "results/external").glob("external_rows_*.jsonl")):
        seen = {json.loads(line)["image_id"] for line in path.read_text().splitlines() if line.strip()}
        exposed.update(seen)
        files.append({"path": str(path.relative_to(root)), "sha256": sha256(path), "n_ids": len(seen)})
    for path in sorted((root / "results/external").glob("external_check_*.json")):
        seen = {r["image_id"] for r in read_json(path).get("rows", [])}
        exposed.update(seen)
        files.append({"path": str(path.relative_to(root)), "sha256": sha256(path), "n_ids": len(seen)})
    sample = set(read_json(root / "data/public_eval/manifest.json")["sample"])
    canonical = json.dumps(derived, sort_keys=True, separators=(",", ":")).encode()
    return {"n_images": len(ids), "schema_columns": columns,
            "native_positive_support": dict(Counter(r for row in derived for r in row["recorded_positive_rule_ids"])),
            "any_recorded_violation": sum(r["any_recorded_violation"] for r in derived),
            "no_recorded_violation": sum(not r["any_recorded_violation"] for r in derived),
            "derived_label_sha256": hashlib.sha256(canonical).hexdigest(),
            "derivation": "native positive fields to IDs; Boolean OR; no finer labels or negatives",
            "known_exposed_ids": sorted(exposed & ids), "historical_files": files,
            "known_exposed_count": len(exposed & ids), "historical_sample_overlap": len(sample & exposed),
            "exposure_audit_scope": "local external_rows/external_check artifacts; absence is not proof of non-exposure",
            "prediction_scores_reused": False, "negative_or_applicability_semantics_established": False}
