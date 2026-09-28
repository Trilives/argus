# Data

## Included here

- `rules/rules_en.json` — the instantiated rule library (42 rules).
- `rules/rules_schema_en.json` — the JSON Schema for a rule unit.
- `gold_images/` — the 500 gold photographs with workers' faces blurred (v2.2.0, CC BY 4.0); see
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
| Label per pair | evidence-based status (compliant / non-compliant / need-review) |
| Reliability (100-image double-annotated subset) | Cohen's κ = 0.948; exact rule-set match = 0.77; mean Jaccard = 0.906 |
| Freeze | fixed under SHA-256 before the reported runs |

**Headline results reproduced from this gold set** (see `../results/` for the machine-readable
summaries): retrieval hit@3 ladder 0.560 / 0.622 / 0.734 / 0.802 / 0.848 (R1 / SigLIP-2 / BM25 /
R3 / R4); best end-to-end GV-F1 0.514; oracle-rule judgement GV-F1 0.867.

The 40 images in `../examples/sample_images/` are a **disjoint, anonymised demonstration set**
(no gold labels); their checksums are in that folder's `manifest.csv` and the blur audit of
P021–P040 in `anonymisation.json`.
