# Manuscript and repository alignment

The manuscript studies applicability-aware provision selection. The company-owned photographs
are a validation platform, not a claimed dataset contribution. The provision library, labels,
judge and R4 agent serve as established inputs/baselines; the atom passes, structured score,
selection modes and evaluation design support the method contribution.

| Claimed contribution | Implementation | Evidence |
|---|---|---|
| Applicability-aware selector at a stated width or miss risk | [RCASR](../src/rcasr.py), [atom posteriors](../src/atom_posterior.py), [typed formulas](../src/typed_schema.py) | [Primary plan](research/design/RCASR_PRESPEC.md) |
| Site-grouped, matched-width evaluation | [Evaluation](../experiments/retrieval/eval_rcasr.py), [end-to-end judge](../experiments/retrieval/judge_rcasr.py) | [Results index](../results/README.md) |
| Failure analysis traced from evidence to selection | [Revision cases](../experiments/retrieval/eval_revision_cases.py) | [Aggregate cases](../results/2026-10-24_revision_posthoc/cases_results.json) |

## Validation populations

The fixed 500-image gold has 880 recorded provision/status pairs. Site identity is recoverable
for 346 images from 48 sites; these supply the primary site-grouped evaluation (601 recorded
pairs, including 509 violations). The other 154 images appear in a separate analysis. Labels
are positive and unlabelled rather than exhaustive; an unrecorded provision is not a confirmed
negative. Site-held-out cross-fitting is not an independent development-unexposed test set.

The additional 500 unlabelled originals and the 40 demonstration files provide supporting
research/example inputs. They do not add gold labels, replace scored pixels or establish new
performance results. ConstructionSite-10k is an auxiliary evaluation with gated third-party data.

## Interpretation and deployment

The included label-only score of 0.514 is the ARGus baseline GV-F1, not the primary RCASR
recorded-violation recall of 0.709. [Results overview](RESULTS_OVERVIEW.md) distinguishes endpoints.
The width-2 comparison was specified after the primary results/retrieval diagnostic but before
width-2 end-to-end scoring; revision H1–H8 analyses are post hoc.

Robot execution demonstrates feasibility and operating proxies, not independently annotated
field accuracy or a separate robot contribution. The full-precision 9B chain passed the stated
width-2 gates. Its robot FP8 checkpoint did not establish recall non-inferiority; a deployment
should verify recall on its own labelled images before relying on that operating point.
Opening and relational hazards remain documented failure modes. CRC relies on its stated
exchangeability and calibration assumptions, not a guarantee for arbitrary deployment images.

## Access and reuse

The manuscript's 500 gold photographs and label-only annotations retain CC BY 4.0. The
additional files have [per-image terms](../data/image_pool/manifest.csv), including a CC BY-NC
non-commercial default for eligible unpublished photographs and explicit prior-grant exceptions.
Software remains Apache-2.0. See [licensing](LICENSING.md) and [data](../data/README.md).
