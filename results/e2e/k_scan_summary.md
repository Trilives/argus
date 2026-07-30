# Controlled candidate-budget (k) scan

Retriever and judgement mode held fixed (J2 `image_rules`); only the candidate budget varies. Cached predictions re-scored, no re-inference. Bootstrap: 2000 image-level resamples. `Δ vs k=3` is a *paired* bootstrap on the same resampled images.

*Width* is the mean number of provisions actually placed before the judge: a fixed ranker fills its budget, the agent stops early.

## BM25 × J2

| k | width | GV-P | GV-R | GV-F1 | GV-F1 95% CI | Δ vs k=3 | 95% CI | sig. |
|---:|---:|---:|---:|---:|---|---:|---|---|
| 1 | 1.00 | 0.4192 | 0.2402 | 0.3054 | [0.2678, 0.3435] | -0.0376 | [-0.0684, -0.0062] | yes |
| 2 | 2.00 | 0.3325 | 0.3806 | 0.3549 | [0.3265, 0.3822] | +0.0119 | [-0.0056, 0.0287] | no |
| 3 | 3.00 | 0.2723 | 0.4631 | 0.3430 | [0.3199, 0.3649] | ref. | — | — |
| 4 | 4.00 | 0.2384 | 0.5355 | 0.3299 | [0.3097, 0.3496] | -0.0131 | [-0.0259, -0.0008] | yes |
| 5 | 5.00 | 0.2129 | 0.5847 | 0.3121 | [0.2947, 0.3298] | -0.0309 | [-0.0467, -0.015] | yes |

- precision strictly decreasing in k: **True**
- recall strictly increasing in k: **True**
- F1 peaks at **k=2** (0.3549); pre-specified k=3 gives 0.3430

## R4 × J2

| k | width | GV-P | GV-R | GV-F1 | GV-F1 95% CI | Δ vs k=3 | 95% CI | sig. |
|---:|---:|---:|---:|---:|---|---:|---|---|
| 1 | 0.99 | 0.6287 | 0.3994 | 0.4885 | [0.4491, 0.5258] | -0.0255 | [-0.056, 0.0062] | no |
| 2 | 1.83 | 0.4932 | 0.5774 | 0.5320 | [0.5045, 0.5597] | +0.0180 | [0.0055, 0.0302] | yes |
| 3 | 2.24 | 0.4379 | 0.6223 | 0.5140 | [0.4855, 0.5401] | ref. | — | — |
| 4 | 2.27 | 0.4356 | 0.6266 | 0.5139 | [0.486, 0.5408] | -0.0001 | [-0.0033, 0.0039] | no |

- precision strictly decreasing in k: **True**
- recall strictly increasing in k: **True**
- F1 peaks at **k=2** (0.5320); pre-specified k=3 gives 0.5140

Note: the F1-optimal k is read off the evaluation set itself, so it is a description of this benchmark, not a validated operating point. Every main-table cell stays at the pre-specified k=3.
