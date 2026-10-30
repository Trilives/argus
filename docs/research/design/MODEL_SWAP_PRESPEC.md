# Model-swap pre-specification: RCASR run end to end on one model

Written 2026-09-24, before any inference for this experiment. Do not edit after the first
`--infer`. Requested by the user on 2026-09-24:

> The framework can run entirely on 9B or entirely on 27B. Describe the 27B as the
> performance baseline, and the 9B as the current on-device trade-off.

This experiment measures that trade-off instead of asserting it. It is run on the local RTX 4090
(48 GB), which is also the GPU the user names as a feasible robot upgrade.

## Question

Take the reported low-width operating point: RCASR-BM25 at mean width 2. The extractor, the scene
descriptions and the judge are all replaced by one model. What happens to screening quality, and
what does that model cost per frame on a 48 GB GPU?

## Arms

| arm | scene descriptions (BM25 prior, judge input) | atom passes | judge | source |
|---|---|---|---|---|
| **REF** (reported) | Qwen3.6-35B-A3B-FP8 | Qwen3.5-9B bf16 | Qwen3.8-27B-FP8 | frozen: `results/2026-09-22_rcasr_lowwidth/`, `results/2026-09-22_rcasr_judge/` |
| **Q9** | Qwen3.5-9B bf16 | Qwen3.5-9B bf16 | Qwen3.5-9B bf16 | new |
| **Q9F** | Qwen3.5-9B-FP8 | Qwen3.5-9B-FP8 | Qwen3.5-9B-FP8 | new; the checkpoint that fits the robot's 16 GB GPU |
| **Q27** | Qwen3.8-27B-FP8 | Qwen3.8-27B-FP8 | Qwen3.8-27B-FP8 | new |

- **Q9 atom evidence.** Q9 reuses the frozen 9B evidence `results/2026-09-22_rcasr_evidence`. It is
  the same checkpoint and config with greedy decoding, so a re-run would reproduce it.
- **Q9F checkpoint.** `Model/Qwen3.5-9B-FP8` is a block-wise (128×128) FP8 quantisation of the same
  weights, the one the robot's existing service runs.

## What is frozen (identical in every new arm)

- **Unchanged from rcasr_v1:** the prompts, the library and overlay, the subject and state question
  sets, the decoding settings, the grid, folds, seeds and bootstrap settings.
  - Decoding: temperature 0, thinking off, top-5 log-probs, pixel bounds.
  - The per-arm configs `data/rules/proposed/rcasr_v1_ms_{q9,q9f,q27}.json` differ from
    `rcasr_v1.json` only in `extractor`, `judge`, `status`, `plan_path` and `version`.
- **Scene descriptions:** the Stage-1 generic `agent_grep` prompt, as in the frozen 35B cache
  and in E5b.
- **Judge:** J3-sym (`rcasr_experiment.judge_pair`), with the arm's own scene descriptions as
  its facts input.
- **Constrained JSON decoding:** on for both the descriptions and the judge. The judge freeze
  already uses it; for the descriptions it is a disclosed difference from the 35B cache.
- **Population:** the 346 site-assigned images, with the same five site folds. The script asserts
  that the folds equal the v2 freeze's.
- **Fitting:** per arm, the `rcasr_bm25` cross-fit is repeated on the arm's own atoms and BM25
  priors at w\* = 3. The grid is unchanged and there is no new tuning. The per-fold threshold is
  then re-set at mean width 2 on the fitting folds, exactly as in `LOWWIDTH_E2_PRESPEC.md`.
- **Failed descriptions:** a description that fails to parse gives an empty query, and so the
  floor prior (the E5b convention). The count is reported.
- **Judge coverage:** the arm's judge judges every pair selected by the arm's RCASR at width 2,
  its RCASR at width 3, and its BM25 at width 3.

## Endpoints and comparisons

The primary endpoints are the grounded-violation recall (GV-R) and the false alarms per image
(FA/img) of each arm's RCASR-BM25 at width 2, on held-out cross-fitted predictions. The judge's
`need_review` share is secondary.

**C1. Does the low-width claim replicate within each arm?** RCASR-BM25 at width 2 is compared with
the same arm's BM25 at width 3. It uses the paired site-unit bootstrap (2000 draws, seed 20260922)
and the gates of the low-width E2:
- W1: the recall CI lower bound is above −0.03.
- W2: the FA/img CI upper bound is below 0.
The claim replicates in an arm if both pass.

**C2. What does the model choice cost?** Each new arm's width-2 arm is compared with REF's width-2
arm. The output is ΔGV-R and ΔFA/img with a paired site-unit bootstrap; the two arms have
different verdict sets, and a unit's statistic uses each arm's own verdicts. There is no gate. The
wording is fixed now:
- CI of ΔGV-R contains 0: "no detectable recall difference from the reference".
- CI entirely below 0: "a recall cost of |Δ|".
- CI entirely above 0: "a recall gain of Δ".
- FA/img is worded the same way.

**C3. Retrieval-level P1 (secondary).** RCASR-BM25 against the matched BM25 prior at width 3 in
each arm, with the same gate as the main analysis.

**Cost proxies (descriptive, on this RTX 4090 48 GB):**
- Model weights: size on disk and GPU memory as loaded, from the vLLM log. vLLM pre-allocates its
  KV pool, so the process total reflects `--gpu-memory-utilization`, not need, and is not reported
  as a requirement.
- **16 GB feasibility (Q9F only).** For the latency bench, the Q9F server is restarted with
  `--gpu-memory-utilization 0.31` (≈ 15.2 GB of the 48 GB card), standing in for the robot's
  16 GB GPU. The bench passes if all 30 frames complete with no error. The bulk Q9F run uses the
  9B serve settings (0.90); greedy outputs do not depend on the memory cap.
- Mean prompt and completion tokens per image.
- Per-frame wall-clock on 30 images drawn with seed 20260922 from the 346, processed one frame at a
  time the way the robot does:
  - the three extraction calls run concurrently;
  - then the selected pairs are judged concurrently;
  - p50 and p95 are reported.

## Interpretation fixed in advance

- **Where the paper's reference stays:** REF remains the reference in the paper. Q27 shows the
  single-model upper configuration; Q9 and Q9F show the on-device configuration.
- **What the arms are:** the new arms are pre-specified additional arms. They do not replace
  any reported number.
- **If C1 fails in Q9F:** the paper says the width-2 workload reduction was not shown for the
  on-device checkpoint.
- **What is not claimed:**
  - On-robot latency. This GPU is a desktop card; the robot figure stays the stored-round proxy
    table.
  - Equivalence, where C2 is inconclusive.

## Limits known now

- **Development exposure:** all images are development-exposed, so this is retrospective
  cross-fitting, not an independent test.
- **Q27 and REF share the judge checkpoint.** Their difference is the descriptions and the atoms.
- **The 9B-FP8 checkpoint is locally quantised,** not a vendor release.
- **Serving stack differences:** the 27B runs on the local copy, not the endpoint. Its rows carry
  `server`, as in `judge_rcasr_shard.py`.

## Execution

Script: `experiments/retrieval/eval_model_swap.py`. The stages are
- `--arm A --prepare`: evidence and description freeze;
- `--infer`, `--fit`, `--judge`;
- `--latency`;
- `--score` (all arms).

Output goes to `results/2026-09-24_model_swap/<arm>/`, and the result record to
`docs/Records/2026-09-24/model_swap.md`. The serve commands are in `Machine.md`; only one model is
served at a time.
