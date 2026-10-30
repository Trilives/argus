# Repository guide

[Documentation index](README.md) · [Data card](../data/README.md) · [Example guide](../examples/README.md)

## Components

| Location | Responsibility |
|---|---|
| `src/` | ARGus screening, symbolic judgement and RCASR selection |
| `Prompts_en/` | English runtime prompts by pipeline stage |
| `configs/` | Frozen robot/runtime configurations |
| `data/rules/`, `data/rule_assets/` | Rule library, typed schema, compiled rule units and retrieval index |
| `data/gold_images/` | Existing evaluation-gold photographs; unchanged |
| `data/research_images/` | Additional privacy-reviewed, unlabelled originals |
| `data/image_pool/` | Combined original-image hashes, subsets and aliases |
| `examples/` | Demonstration files and inference guide |
| `docs/research/` | Grouped index of the immutable analysis plans |
| `experiments/retrieval/` | Analysis and evaluation entry points |
| `results/` | Aggregate tables, input snapshots and redaction notes |
| `reproduce/` | Label-only gold/predictions, scorer, snapshot exceptions and release manifest |
| `scripts/verify_release.py` | Offline integrity, metadata, navigation and snapshot verification |

## RCASR implementation

```
src/
├── rcasr.py                 # structured applicability score + width / CRC selection
├── atom_posterior.py        # token-level atom posteriors with abstention (two rule-agnostic passes)
├── typed_schema.py, typed_predicates.py, typed_evidence.py   # provision formulas compiled over atoms
├── rcasr_experiment.py      # site-grouped cross-fitting, arms, ablations, gates, CRC calibration
├── site_partition.py        # site units and folds (near-duplicate clusters fused)
├── set_selection*.py        # set-selection metrics and matched-width reporting
├── public_map.py            # frozen mapping of ConstructionSite-10k rules to the library
└── research_snapshot.py     # SHA-256 input snapshots and their verification
experiments/retrieval/
├── eval_rcasr.py            # internal evaluation (evidence → cross-fitted selection)
├── judge_rcasr.py, judge_rcasr_shard.py   # frozen end-to-end judge over the selected sets
├── eval_lowwidth_e2.py      # the pre-specified width-2 arm
├── eval_rcasr_public.py     # public auxiliary evaluation (ConstructionSite-10k)
├── eval_s34_default.py      # the no-subject abstention default adopted by the deployed judge
├── eval_model_swap.py       # the whole chain on one model
├── eval_rcasr_tsr.py        # RCASR on a two-stage retrieval prior (src/two_stage_prior.py)
└── eval_revision_posthoc.py, eval_revision_cases.py   # revision post-hoc analyses H1–H8 over frozen outputs
configs/
├── rcasr_runtime_v1.json              # the frozen configuration the robot runs (sha256 ea624e49…)
├── rcasr_runtime_v1_recal_site.json   # its label-free, site-recalibrated threshold (sha256 4cad9da1…)
└── rcasr_runtime_v1_recal_site_q9f.json  # the same, one FP8 checkpoint as extractor and judge (16a14292…)
data/rules/proposed/rcasr_v1.json      # frozen RCASR hyper-parameters and pre-specified grid
data/rules/proposed/typed_evidence_v1.json   # the 131-atom vocabulary and per-provision bindings
data/rule_assets/                      # rule units and the BM25 index built from the library
data/public_eval/public_map_v1.json    # public rule mapping (the public labels are not redistributed)
docs/research/design/*.md              # analysis plans, each written before its run
results/2026-*/                        # aggregate results + input snapshots for every v3 table
```


## ARGus framework


The screening framework RCASR plugs into. Given a site photograph, a bounded retrieval **agent**
proposes the few provisions that plausibly apply. The verdict is then computed from a named provision
formula over persisted checkpoint evidence, never the agent's free text, so every decision traces
back to a clause. The framework measures where screening fails: handed the correct
provision, the judge reaches a grounded-violation F1 of **0.867**, while the best complete
configuration reaches **0.514**. Retrieval, deciding which provisions apply, dominated end-to-end
accuracy, and grounded precision fell at every widening step measured
([`results/e2e/k_scan_summary.md`](../results/e2e/k_scan_summary.md)). That finding motivates RCASR.
The framework supplement is [`paper/supplement.pdf`](../paper/supplement.pdf).


## Rule units


A regime is onboarded by decomposing it into rule units of one schema — not by retraining.
Each unit carries provision provenance (`source_level`, `source_quote`), atomic
`visual_checkpoints`, a boolean `visual_screening_rule` over those checkpoints, a
`decision_scope` (so quantities beyond a single image route to review rather than being
guessed), retrieval text, and `rectification_advice`. See `data/rules/rules_schema_en.json`.
