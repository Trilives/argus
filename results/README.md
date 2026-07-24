# Results (aggregate)

Machine-readable summaries behind the paper's tables. These are **aggregate** metrics computed on
the frozen 500-image gold set (see `../data/README.md`); no images, gold labels, or per-image
outputs are included.

| File | What it backs |
|---|---|
| `retrieval/annotated_retrieval_summary.json` | Retrieval ladder — hit@k / recall@k for the fixed and agentic retrievers (R1–R4, BM25, SigLIP-2). |
| `judgement/oracle_grounded_summary.json` | Oracle-rule judgement (RQ2) — the retrieval-free upper bound (GV-F1 0.867). |
| `e2e/grounded_violation_summary.json` | End-to-end Grounded-Violation P/R/F1 for the full retriever × judge matrix (RQ3). |
| `e2e/grounded_ci_summary.json` | Paired image-level 95% confidence intervals and prespecified contrasts. |
| `e2e/e2e_multi_instance_summary.json` | Multi-instance end-to-end scoring (the task's natural form). |
| `e2e/no_retrieval_summary.json` | No-retrieval control (candidate selection as a precision lever). |
| `e2e/compliant_fp_summary.json` | Compliant-image false-positive exposure vs candidate width. |
| `e2e/risk_coverage_summary.json` | Risk–coverage / selective-prediction behaviour. |

Metric definitions and the analysis are in the manuscript; regenerate these with the evaluation
harness in the full research repository (available with the dataset on request).
