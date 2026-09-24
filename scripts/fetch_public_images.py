#!/usr/bin/env python3
"""Fetch ConstructionSite-10k test images for the RCASR guide, with your own gated access.

    export HF_TOKEN=<read token of the account that accepted the dataset's access terms>
    uv run --group analysis python scripts/fetch_public_images.py            # the 8 guide images
    uv run --group analysis python scripts/fetch_public_images.py --ids 0000007 0000039

ConstructionSite-10k (Chen and Zou, CC BY-NC 4.0; images from the MOCS dataset) is gated on
Hugging Face, so this repository never redistributes its images. The script downloads the test
split with your token, writes only the requested images to ``examples/public_images/``
(git-ignored), and checks every image's SHA-256 against the public run's input snapshot, so you
screen byte-identical inputs. Non-commercial research use only; see the dataset card.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "LouisChen15/ConstructionSite"
SNAPSHOT = ROOT / "results/2026-09-22_rcasr_public/input_snapshot.json"
OUT = ROOT / "examples/public_images"

# Test images never processed in earlier work: two per public rule where RCASR's frozen
# selection let the judge confirm a recorded violation that matched-width BM25 missed
# (rule_1 PPE, rule_2 harness at height, rule_3 edge protection), and two excavator-radius
# images (rule_4) where RCASR drops the provision, the failure the paper reports.
GUIDE_IDS = {
    "0000068": "rule_1 (PPE): detected by RCASR, missed by BM25",
    "0000095": "rule_1 (PPE): detected by RCASR, missed by BM25",
    "0000328": "rule_2 (harness at height): detected by RCASR, missed by BM25",
    "0001392": "rule_2 (harness at height): detected by RCASR, missed by BM25",
    "0000009": "rule_3 (edge protection): detected by RCASR, missed by BM25",
    "0000626": "rule_3 (edge protection): detected by RCASR, missed by BM25",
    "0000327": "rule_4 (excavator radius): the reported failure, missed by RCASR",
    "0000331": "rule_4 (excavator radius): the reported failure, missed by RCASR",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--ids", nargs="+", default=sorted(GUIDE_IDS), help="test image ids")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN; request access first at https://huggingface.co/datasets/" + REPO_ID)
    expected = {i: row["sha256"] for i, row in json.loads(SNAPSHOT.read_text())["images"].items()}
    unknown = sorted(set(args.ids) - set(expected))
    if unknown:
        raise SystemExit(f"not in the public evaluation (412 test images): {unknown}")

    shard = hf_hub_download(REPO_ID, "test.parquet", repo_type="dataset", token=token)
    OUT.mkdir(parents=True, exist_ok=True)
    wanted, written = set(args.ids), {}
    handle = pq.ParquetFile(shard)
    for group in range(handle.num_row_groups):  # a row group at a time; the shard is >1 GB
        for row in handle.read_row_group(group, columns=["image_id", "image"]).to_pylist():
            if row["image_id"] not in wanted:
                continue
            payload = row["image"]["bytes"]
            if hashlib.sha256(payload).hexdigest() != expected[row["image_id"]]:
                raise SystemExit(f"{row['image_id']}: bytes differ from the public run's snapshot")
            (OUT / f"{row['image_id']}.jpg").write_bytes(payload)
            written[row["image_id"]] = GUIDE_IDS.get(row["image_id"], "")
    missing = sorted(wanted - set(written))
    if missing:
        raise SystemExit(f"not found in the test split: {missing}")
    for image_id, note in sorted(written.items()):
        print(f"  {image_id}.jpg  {note}")
    print(f"\n{len(written)} image(s) in {OUT.relative_to(ROOT)}/, hashes match the public run.")
    print("ConstructionSite-10k by X. Chen and Z. Zou, CC BY-NC 4.0; images from the MOCS dataset.")


if __name__ == "__main__":
    main()
