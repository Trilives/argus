# Reproducing the evaluation

This folder lets a reviewer verify the central claims **without the private images
or annotations** — using a cryptographic freeze proof, the label-only gold, and a
self-contained scorer.

## 1. The gold set was frozen before the results (pre-registration)

`freeze_manifest.json` records the SHA-256 of the annotation and rule files at a
timestamp:

```json
{"status": "frozen", "created_at": "2026-07-16T13:22:06Z",
 "file_sha256": {"…/image_rule_gold.json": "c5674b…", "…/rules_en.json": "1cfd2d…", …}}
```

Any gold file we later provide on request must hash to these values, which proves
the labels and rule set were fixed *before* the reported runs and were not altered
afterward. Verify with `sha256sum <file>` against the manifest.

## 2. The frozen annotations (label-only)

`gold_labels.json` is a faithful **label-only projection** of the frozen gold —
`{image_id: {rule_id: status}}`, no images and no scene text. It is the exact gold
used to score every reported number. Reproduce the dataset counts:

```bash
python -c "import json; g=json.load(open('gold_labels.json')); \
pairs=sum(len(v) for v in g.values()); \
from collections import Counter; d=Counter(s for v in g.values() for s in v.values()); \
print('images', len(g), 'pairs', pairs, dict(d))"
# -> images 500 pairs 880 {'non_compliant': 691, 'compliant': 174, 'undetermined': 15}
```

(The manifest in §1 hashes the *original* gold file that is available on request;
`gold_labels.json` is the label-only view of it.)

## 3. Recompute the headline metric (worked example)

`score_grounded_violations.py` is a self-contained port of the paper's scorer.
`predictions/R4_image_rules.jsonl` is the **label-only** prediction file for the best
end-to-end system (R4 image-conditioned retrieval × J2 image-rules judgement). Run:

```bash
python score_grounded_violations.py gold_labels.json predictions/R4_image_rules.jsonl
```

reproduces the paper's headline cell exactly, from released files alone:

```
gv_tp 430 · gv_pred 982 · gv_gold 691
GV precision (micro) 0.4379 · recall 0.6223 · F1 0.514
GV precision (macro) 0.4698 · recall 0.6529
```

Each prediction row is `{image_id, violation_instances: [{rule_id, compliance_label}]}`
(the scorer also accepts `{image_id, model_label, matched_rule_id}`) — regenerate it by
running the pipeline (`../run_demo.py` or the research harness) and keeping only those
label-only fields. Prediction files for the other retriever × judge cells are available
on request.

## 4. Reliability of the gold labels

`reliability_agreement.json` is the authoritative inter-annotator report for the
100-image double-annotated subset (20 % of the gold, seed `20260713`), comparing the
**raw primary** annotator with a second annotator (`anno2`):

```
applicable-set exact match  0.77
mean Jaccard                0.906
status Cohen's κ            0.948   (raw agreement 0.982 over 171 shared status pairs)
disagreements               29  = 14 rule-only-primary + 12 rule-only-anno2 + 3 status conflicts
```

The file is label-only (`{image_id, rule_id, kind, primary, anno2}`) and includes the
full 29-item disagreement list, so the top-line numbers are inspectable rather than
asserted. Note these are computed on the **raw, pre-arbitration** labels; the
`gold_labels.json` in this folder is the *post-arbitration* final gold, so it is the
correct input for the metric (§3) but **not** for recomputing annotator agreement. The
raw two-annotator label files for a from-scratch agreement recomputation are available
on request under the data-use agreement.

## Not included (privacy / site-operator agreement)

Raw site images, cached scene-fact records, and per-checkpoint evidence text describe
identifiable people and scenes; they are available from the authors on reasonable
request under a data-use agreement. Everything needed to audit the *claims* — the
freeze proof, the label-only gold, the scorer — is here.
