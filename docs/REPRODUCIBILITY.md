# Reproducibility and frozen inputs

Start with [the release verifier](../scripts/verify_release.py), [the label-only reproduction guide](../reproduce/REPRODUCE.md) and [the analysis-plan index](research/README.md).

```bash
uv run --locked python scripts/verify_release.py
uv run --locked --with pytest python -m pytest -q tests
uv run --locked python reproduce/score_grounded_violations.py reproduce/gold_labels.json reproduce/predictions/R4_image_rules.jsonl
```

## Research snapshot preservation

Each `results/2026-*/input_snapshot.json` records its original input hashes. The
[path-migration map](../reproduce/path_migrations.json) resolves relocated plans without changing
their contents. Code-reference edits are recorded with before/after hashes and verified as path
substitutions; the snapshot resolver handles those exact substitutions. Model logic, frozen
configurations, rule assets and result payloads are preserved. Restricted inputs and intentional
public-package differences have explicit explanations:

- **Restricted inputs are not released:** the unblurred images (the released ones in
  `data/gold_images/` are blurred and scaled, so their hashes differ), the full gold records, site
  keys, scene facts, atom evidence, per-pair verdicts, the per-frame reports of the robot round and the
  ConstructionSite-10k labels.
- **Files belong to this public package, not to the research checkout:** `pyproject.toml`,
  `uv.lock`, and `src/config.py` / `src/schemas.py` (comments only).
- **Redacted:** `results/2026-09-22_rcasr_v2/rcasr_results.json`,
  `results/2026-09-27_rcasr_tsr/tsr_results.json` and
  `results/2026-10-24_revision_posthoc/cases_results.json` (see `results/README.md`). The two-stage run's
  snapshot collapses its 500 image entries into one digest; its other hashes are unchanged.
- **Re-serialised JSON:** `results/data_audit/near_duplicates.json` has the same
  content in different formatting.
- **Skip guard added:** `tests/test_site_partition.py` skips when the restricted site keys are
  absent.

`configs/rcasr_runtime_v1.json` names the snapshots it was derived from, and those hashes verify
against the files here.

**Tables.** `results/README.md` maps every table of the paper to the file behind it. The internal
result `results/2026-09-22_rcasr_v2/rcasr_results.json` is released with construction-project names
replaced by `site_NN` and lost gold pairs reduced to counts, so its hash differs from the research
copy.


## Public-package exceptions

The original snapshots are retained without rehashing. [snapshot_exceptions.json](../reproduce/snapshot_exceptions.json) records every absent frozen input and every intentionally different public file, with reasons and hashes of the public copies. The verifier rejects any new unexplained difference.

Analysis-plan payloads, model logic, configurations, image pixels and archived results are preserved. Documentation relocations and their code references are recorded in `reproduce/path_migrations.json`; stored snapshots are not rehashed. New data are an unlabelled research extension and do not replace experimental inputs. Blurring and resizing mean fresh model runs on released JPEGs cannot exactly reproduce predictions on the private originals. Label-only gold and predictions reproduce the reported score directly.

## Running the included code

The [example guide](../examples/README.md) documents inference on bundled files and fetching third-party guide images with the user's own gated access. Most historical experiment entry points require restricted inputs listed in their snapshots; the public package does not claim those runs are independently executable without those inputs. Model weights and private site keys are not silently replaced with different data.

Eight tests skip when private site keys or gated ConstructionSite-10k labels are unavailable. The full pytest command above discovers both function-style and unittest-style tests. No model inference, account credentials or model downloads are needed for the default test suite.
