# Results (aggregate)

Machine-readable summaries behind the paper's tables, computed on the frozen 500-image gold set
(see `../data/README.md`). These are **aggregate** metrics: no images, no gold labels, and no
per-pair model outputs are included. The one exception is the near-duplicate audit, which lists bare
image identifiers — no labels, no imagery — and only for the perceptual-hash clusters it reports.

Regenerate with `scripts/mirror_public_results.py` in the full research repository.

## Retrieval

| File | What it backs |
|---|---|
| `retrieval/annotated_retrieval_summary.json` | Retrieval ladder — hit@k / recall@k for the fixed and agentic retrievers (R1–R4, BM25, SigLIP-2). |
| `retrieval/rq1_metric_table.{json,md}` | The full ladder including the cost-matched non-agent controls (cross-encoder, one-shot VLM selector), with precision@k — the column showing that width, not agency, separates the arms. |
| `retrieval/rrf_fusion_summary.{json,md}` | Reciprocal-rank fusion of BM25/R1 with R4: recall is purchasable at the retrieval level (hit@5 0.942). |
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
