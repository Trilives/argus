# Repository verification

From the repository root:

```bash
uv lock --check --offline
uv run --locked python scripts/verify_release.py
uv run --locked --with pytest python -m pytest -q tests
uv run --locked python scripts/bootstrap_rule_library.py --check data/rules/rules_en.json
uv run --locked python reproduce/score_grounded_violations.py \
  reproduce/gold_labels.json reproduce/predictions/R4_image_rules.jsonl
```

The offline verifier checks the original and demonstration image manifests, JPEG metadata,
subset membership, anonymous source-byte aliases, processing audits, active documentation
links, package metadata and every public-file hash.
It also checks the per-image licence assignments, non-commercial scope and prior-grant exceptions.

[Research plans](research/README.md) are organized by study stage. The
[path-migration record](../reproduce/path_migrations.json) preserves the original plan hashes
and records their current locations. Original snapshots are not rewritten. The
[snapshot exceptions](../reproduce/snapshot_exceptions.json) explain restricted inputs and
intentional public-package differences.

The [label-only scorer](../reproduce/REPRODUCE.md) returns GV-F1 0.514: 430 true positives,
982 predictions and 691 recorded gold violations. Tests requiring private site keys or
gated third-party labels skip explicitly. Inference examples use the user's configured
model server and are separate from offline verification.
