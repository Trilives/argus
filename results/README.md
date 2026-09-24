# Results (aggregate)

Machine-readable summaries behind the paper's tables, for the earlier study computed on the frozen 500-image gold set
(see `../data/README.md`). These are **aggregate** metrics: no images, no gold labels, and no
per-pair model outputs are included. The one exception is the near-duplicate audit, which lists bare
image identifiers — no labels, no imagery — and only for the perceptual-hash clusters it reports.

Regenerate with `scripts/mirror_public_results.py` in the full research repository.

## RCASR (v2.0.0): the tables of the current paper

Each run directory keeps its research-repository path and carries the `input_snapshot.json` that
pins its inputs by SHA-256. The analysis plans are in `../docs/paper_v3/design/`.

| File | What it backs |
|---|---|
| `2026-09-22_rcasr_v2/rcasr_results.json` | Internal evaluation (346 images, 48 site units, five site-grouped folds): the pre-specified gates, every arm and one-factor ablation at matched width (BM25 and agent priors), CRC calibration, per-site transfer, leave-one-family-out and harm accounting. Site names are replaced by `site_NN`, and lost gold pairs are reduced to counts. |
| `2026-09-22_rcasr_judge/e2e_results.json` | End-to-end screening with one frozen judge at width 3: grounded-violation recall/precision/F1, false alarms and review load per arm. |
| `2026-09-22_rcasr_lowwidth/lowwidth_results.json` | The pre-specified width-2 arm against BM25 at width 3: non-inferior recall, false alarms per image 1.28 → 1.00, empty-set rate, token cost. |
| `2026-09-22_rcasr_public/public_results.json` | ConstructionSite-10k auxiliary evaluation (412 test images with a recorded violation): detection per public rule. |
| `2026-09-22_rcasr_public/final_config.json` | The configuration frozen on the internal images before public scoring. |
| `2026-09-22_rcasr_public/public_selection.json` | RCASR's and matched BM25's selected provisions for each public test image. |
| `2026-09-23_s34_default/s34_results.json` | The no-subject abstention default: identical `non_compliant` sets under both rules on every judged pair, and the review-load change. |
| `2026-09-23_deployment/deployment_parity.json` | Runtime parity: the deployed selector reproduces the frozen selections; stated facts-model difference. |
| `2026-09-23_deployment/deployment_simulation.json` | Stored-frame replay through the deployed runtime (pilot): decision rates, realised width and latency per slice. |
| `2026-09-23_deployment/onrobot_round.json` | The on-robot round under the frozen and the site-recalibrated threshold: the proxies of the robot table. Unlabelled, so there are no accuracy figures. |

The runtime configurations the robot runs are in `../configs/`.

## ARGus (v1.x): the tables of the earlier study

## Retrieval

| File | What it backs |
|---|---|
| `retrieval/annotated_retrieval_summary.json` | Retrieval ladder — hit@k / recall@k for the fixed and agentic retrievers (R1–R4, BM25, SigLIP-2). |
| `retrieval/rq1_metric_table.{json,md}` | The full ladder including the cost-matched non-agent controls (cross-encoder, one-shot VLM selector), with precision@k — the column showing that width, not agency, separates the arms. |
| `retrieval/rrf_fusion_summary.{json,md}` | Reciprocal-rank fusion of BM25/R1 with R4: recall is purchasable at the retrieval level (hit@5 0.942) but does not convert end to end — see `e2e/rrf_e2e_summary`. |
| `retrieval/iter_sweep_summary.{json,md}` | Agent iteration-budget sweep (caps 2/4/8/16): mean iterations saturate near 3.3, so the cap is over-provisioned. |

## Judgement

| File | What it backs |
|---|---|
| `judgement/oracle_grounded_summary.json` | Oracle-rule judgement — the retrieval-free upper bound (GV-F1 0.867), which is why judgement is not the binding stage. |
| `judgement/crossmodel_summary.{json,md}` | Cross-model replication (Qwen3.6-35B / gemma-4-12B / Qwen3.5-9B): the judge ordering is stable and barely scale-sensitive. |
| `judgement/crossmodel_ci_summary.{json,md}` | Paired image-level bootstrap on that gap. A point estimate cannot support "indistinguishable", so this is what backs the claim: 9B vs 35B delta-F1 0.010, 95% CI [-0.004, 0.025] — **not** significant; gemma-4-12B 0.044, [0.027, 0.062] — significant. |

## End to end

| File | What it backs |
|---|---|
| `e2e/grounded_violation_summary.json` | Grounded-Violation P/R/F1 for the full retriever × judge matrix. |
| `e2e/grounded_ci_summary.json` | Paired image-level 95% bootstrap confidence intervals and the prespecified contrasts. |
| `e2e/e2e_multi_instance_summary.json` | Multi-instance end-to-end scoring (the task's natural form). |
| `e2e/no_retrieval_summary.json` | No-retrieval control — all 42 provisions as candidates. |
| `e2e/rrf_e2e_summary.{json,md}` | The fusion arms scored end to end: better retrieval recall, worse grounded precision, on every judge model. |
| `e2e/compliant_fp_summary.json` | Compliant-image false-positive exposure against candidate width. |
| `e2e/risk_coverage_summary.json` | Risk–coverage / selective-prediction behaviour. |
| `e2e/split_leakage_summary.{json,md}` | Tuning exposure: cached predictions re-scored on disjoint strata of the frozen gold. |
| `e2e/applicability_gate_summary.{json,md}` | The Kleene three-valued applicability gate: filter rate, filter precision, abstention and coverage per cell. |
| `e2e/k_scan_summary.{json,md}` | Controlled candidate-budget sweep: one retriever cut at several widths with the judge and judgement mode held fixed, so width is separated from the mechanism that produced it. Grounded precision falls strictly and F1 peaks at k=2 for both a lexical and an agentic retriever. The k=2 optimum is read off the evaluation set, so the paper keeps the pre-specified k=3 everywhere; this file carries both. |
| `e2e/error_composition_summary.{json,md}` | Error decomposition at the headline cell: retrieval misses vs judgement misses, and the same-category sibling share behind the sibling-confusion claim. |
| `e2e/repeats_summary.json` | Run-to-run variance over repeat runs of the headline cells (decoding is stochastic; this is the spread the CIs sit on). |

## Data audit

| File | What it backs |
|---|---|
| `data_audit/near_duplicates.{json,md}` | Perceptual-hash near-duplicate rates for the image pool and the gold set, with a threshold sweep. |
| `data_audit/rule_provenance.{json,md}` | Per-provision provenance: governing standard and provenance tag, plus the screening-predicate derivation (transcribed clause vs. engineering-derived proxy, dropped normative fields, decidability scope). Same table as the online supplement. |
| `data_audit/agreement_stratified.md` | Inter-annotator agreement broken down by stratum, showing reliability is limited by provision *applicability* rather than compliance status. |

## External comparisons

| File | What it backs |
|---|---|
| `external/detector_baseline_summary.{json,md}` | Head-to-head against a supervised hard-hat detector on the helmet provision, over all 500 images. |
| `external/rectification_eval_summary.{json,md}` | Rated relevance of the generated rectification advice (two raters, 50 items). |

Metric definitions and the analysis are in the manuscript. The evaluation harness ships with the full
research repository, available with the dataset on request.
