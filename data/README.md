# Data

These photographs support validation of the method; the paper does not claim a new dataset
contribution. The [paper-alignment guide](../docs/PAPER_ALIGNMENT.md) distinguishes the scored
population from unlabelled supporting material. See [per-image licences](image_pool/manifest.csv):
463 additional files are non-commercial only, with existing grants retained for the others.

## Included here

- `rules/rules_en.json` — the instantiated rule library (42 rules).
- `rules/rules_schema_en.json` — the JSON Schema for a rule unit.
- `gold_images/` — the 500 gold photographs with workers' faces blurred (CC BY 4.0); see
  [`gold_images/README.md`](gold_images/README.md). Their labels are
  [`../reproduce/gold_labels.json`](../reproduce/gold_labels.json).

## Evaluation gold set — data card

The quantitative results were computed on the frozen gold set below. Its photographs (faces
blurred) and its label-only annotations are included. The unblurred photographs, the full gold
records with their evidence text, and the site keys are not distributed; they are available from the
authors on reasonable request under a data-use agreement.

| Property | Value |
|---|---|
| Images | 500 (construction sites, Shenzhen regime) |
| Annotated image–rule pairs | 880 |
| Rules | 42, in 4 categories (OPN opening, EDG edge, BHV behaviour/PPE, CIV area) |
| Label per pair | evidence-based status (`compliant` / `non_compliant` / `undetermined`) |
| Reliability (100-image double-annotated subset) | Cohen's κ = 0.948; exact rule-set match = 0.77; mean Jaccard = 0.906 |
| Freeze | fixed under SHA-256 before the reported runs |

The primary RCASR evaluation uses the 346 site-assigned images from 48 sites (601 recorded
pairs, including 509 violations). The 154 images without site identity are treated separately.
Labels are not exhaustive; unrecorded provisions are not confirmed negatives.

**Headline results reproduced from this gold set** (see `../results/` for the machine-readable
summaries): retrieval hit@3 ladder 0.560 / 0.622 / 0.734 / 0.802 / 0.848 (R1 / SigLIP-2 / BM25 /
R3 / R4); best end-to-end GV-F1 0.514; oracle-rule judgement GV-F1 0.867.

The 40 images in `../examples/sample_images/` are a **disjoint, anonymised demonstration set**
(no gold labels); their checksums are in that folder's `manifest.csv` and the blur audit of
P021–P040 in `anonymisation.json`.

## Expanded original-image pool

An additional 500 reviewed, unlabelled originals are included in [research_images/](research_images/README.md). The combined original-image pool contains 1,000 IDs; the evaluation gold and reported results above are unchanged. No crops are included. The [combined manifest](image_pool/manifest.csv) records raw-input and anonymised-output hashes separately, including source-byte aliases.

Of the additions, 463 use CC BY-NC 4.0, 19 retain CC BY 4.0 gold-image grants and 18 retain
Apache-2.0 demonstration-image grants. Commercial use of the 463 non-commercial files requires
separate written company permission. The [licence](research_images/LICENSE) and
[exceptions](research_images/LICENSE_EXCEPTIONS.csv) define the scope. The 500 gold photographs
and [label-only annotations](../reproduce/DATA_LICENSE) retain the manuscript's CC BY 4.0 terms.
