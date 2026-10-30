# Research plans and experiment map

[Documentation](../README.md) · [Results](../../results/README.md) ·
[Reproducibility](../REPRODUCIBILITY.md) · [Analysis timing](../RESEARCH_DISCLOSURES.md#analysis-timing)

Read the primary plan first, then later comparisons, deployment records and post-hoc analyses.
Plan contents retain their original hashes. The [migration map](../../reproduce/path_migrations.json)
resolves old snapshot paths to this directory.

## 1. Primary study

| Plan | Purpose | Implementation/results |
|---|---|---|
| [Scorer design proposal](design/RCASR_PROPOSAL.md) | Historical motivation and proposed interface; its original approval-status heading remains as history | [RCASR code](../../src/rcasr.py) |
| [Primary RCASR plan](design/RCASR_PRESPEC.md) | Site-grouped width-3 comparisons, asymmetric costs, CRC and public auxiliary evaluation | [Internal](../../results/2026-09-22_rcasr_v2/rcasr_results.json), [end-to-end](../../results/2026-09-22_rcasr_judge/e2e_results.json), [public](../../results/2026-09-22_rcasr_public/public_results.json) |

## 2. Later comparisons and extensions

| Plan | Timing/question | Results |
|---|---|---|
| [Width-2 end-to-end arm](design/LOWWIDTH_E2_PRESPEC.md) | Specified after width-3 results and the width-2 retrieval diagnostic, before width-2 end-to-end scoring | [Width-2](../../results/2026-09-22_rcasr_lowwidth/lowwidth_results.json) |
| [No-subject judgement default](design/S34_DEFAULT_PRESPEC.md) | Reanalysis of existing evidence: abstention and review load | [Default comparison](../../results/2026-09-23_s34_default/s34_results.json) |
| [Single-model chain](design/MODEL_SWAP_PRESPEC.md) | Same chain with alternative extractor/judge checkpoints | [Model swap](../../results/2026-09-24_model_swap/model_swap_results.json) |
| [Two-stage retrieval prior](design/TWO_STAGE_PRIOR_PRESPEC.md) | SigLIP 2 followed by cross-encoder reranking | [Two-stage prior](../../results/2026-09-27_rcasr_tsr/tsr_results.json) |

## 3. Robot deployment and operating proxies

| Plan | Scope | Results |
|---|---|---|
| [Stored on-robot round](design/ONROBOT_ROUND_PRESPEC.md) | Written after the frozen-threshold round, before applying recalibration | [Round](../../results/2026-09-23_deployment/onrobot_round.json) |
| [Chain on the robot GPU](design/ONROBOT_FP8_PRESPEC.md) | Replay of that round with one local FP8 checkpoint | [FP8](../../results/2026-09-27_onrobot_fp8/onrobot_fp8.json) |

These unlabelled records do not establish independently annotated field accuracy.

## 4. Post-hoc revision analyses

| Plan | Scope | Results |
|---|---|---|
| [Revision H1–H8](design/REVISION_POSTHOC_PRESPEC.md) | Explicitly post hoc; original runs are preserved | [Statistics/ablations](../../results/2026-10-24_revision_posthoc/posthoc_results.json), [cases/data/vocabulary](../../results/2026-10-24_revision_posthoc/cases_results.json) |

## 5. Data context

The [expanded original-image pool](../../data/image_pool/README.md) adds unlabelled research
inputs without adding evaluation gold or new experiments. [Snapshot exceptions](../../reproduce/snapshot_exceptions.json)
explain restricted inputs and intentional public-package differences. Snapshots are not rehashed.
