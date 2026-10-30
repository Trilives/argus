# RCASR — pre-specification

**Written:** 2026-09-22, before any RCASR inference or scoring. **Do not edit after the first
run.** Outcomes go in a separate record. Module approval: `RCASR_PROPOSAL.md` (user,
2026-09-22). Frozen configuration: `data/rules/proposed/rcasr_v1.json`.

## Status of the data

All 500 internal images are development-exposed. Everything below is a **retrospective,
site-grouped cross-fitted analysis**, not an independent test. The opening/shaft/edge atom
class (below) was chosen from the subject-screen harm table computed on all 500 images, so
that design choice is not blind to any fold; this is disclosed wherever the result is reported.

## Inputs

- Site folds: `data/splits/site_keys_confirmed.json` (48 sites, 346 images, user decision
  2026-09-22), near-duplicate clusters fused into units, K = 5, seed 20260922
  (`src/site_partition.py`). The 154 images without a site are excluded from every
  cross-fitted endpoint and appear only in a labelled all-500 resubstitution appendix.
- Retriever priors: BM25 over the frozen fact cache (`set_selection_report.load_inputs`,
  $\pi=s/(1+s)$). Cached R4 (`results/retrieval/agent_grep_visual_eval.json`), mapped
  label-free to $\pi = (5-\text{rank})/5$ for ranks 1–4 and $\pi = 0.1$ for unretrieved rules.
- Atom posteriors: Qwen3.5-9B (`Model/Qwen3.5-9B`, served locally by vLLM 0.24.0, greedy,
  thinking off, repetition penalty 1.1, pixel budget as `src/config.py`), top-5 token log-probs
  at each answer token. Two rule-agnostic passes per image:
  1. **subject pass**: the 39 subject-gate questions and template of `subject_screen_v2.json`,
     unchanged;
  2. **state pass**: 92 questions covering every non-external boolean atom that appears in a
     rule's `visual_screening_rule` and not in a gate (`scrap_condition_met` is external-only
     and stays unknown). No prompt names a rule, provision or violation.

For each answer token, $p_{yes}, p_{no}, p_{unc}$ are the summed probabilities of top-5
tokens that are non-empty prefixes of `yes`, `no`, `unclear`, then renormalised. An id with no
answer token, or with zero mass on all three, is **unknown**.

## Method

- Atom posterior $q=p_{yes}/(p_{yes}+p_{no})$. The atom is **known** iff
  $(1-p_{unc})\max(q,1-q)\ge\tau$, with $\tau_{open}$ in place of $\tau$ for the six atoms
  `horizontal_opening_present`, `elevator_shaft_present`, `guardrail_present`,
  `roof_balcony_platform_edge_present`, `shaft_opening_present`, `pit_or_trench_edge_present`.
- Literal probability: $q$ for `== yes` and $1-q$ for `== no`. An unknown atom's literal is
  **1**. The upper bound is therefore taken over unknowns, and an unknown can never lower a
  score (the soft form of "unknown never excludes").
- AND → product, OR → noisy-or. $P_{gate}$ is evaluated over the compiled subject gate and
  $P_{vis}$ over the compiled `visual_screening_rule`. $P_{vis}=1$ for the five rules without
  one.
- Score: $s=\sigma(\operatorname{logit}\pi+\theta_1\log P_{gate}+\theta_2\log P_{vis})$, with
  logs clipped at $\log 10^{-4}$.
- Adaptive selection: a single global threshold per fold, taken on the fitting folds at the
  width budget $w^*$ (the (n·w*)-th largest score) and applied unchanged to the held-out fold.
  Per-image cardinality varies.
- Grids, fixed now: $\theta_1,\theta_2\in\{0,.5,1,2,4\}$, $\tau\in\{0,.5,.7,.8,.9,.95\}$,
  $\tau_{open}\in\{0,.5,.7,.8,.9,.95,.99\}$. The fit maximises recorded-violation recall at
  $w^*$ on the fitting folds. Ties go to the first grid point in lexicographic order
  $(\theta_1,\theta_2,\tau,\tau_{open})$.
- Budgets: $w^*$ is the mean width of the fixed k = 3 reference on the fitting population
  (3.0 for BM25; R4 top-3's realised mean for R4).
- Separation: for held-out fold $f$, the calibration fold is $(f+1)\bmod 5$ and the other three
  are fitting folds. Weights, τ and the width threshold come from the fitting folds only, and
  CRC uses the calibration fold only.

## Endpoints (cross-fitted held-out predictions, 346 site images)

Each endpoint is a paired comparison at the **same total candidate count**. The reference is
`matched_rank_sets(π, W)` at the RCASR arm's realised held-out total `W`. Uncertainty is a
2,000-replicate bootstrap over **site units**.

| id | arm | reference | GO if |
| --- | --- | --- | --- |
| **P1** | `rcasr_bm25` | BM25 rank mixture | Δ recall ≥ +0.03 and CI lower bound > 0 |
| **P2** | `rcasr_r4` | R4 rank mixture | Δ recall ≥ +0.03 and CI lower bound > 0 |
| S1 | `rcasr_bm25` | `adaptive_bm25` (θ₁=θ₂=0, same global-threshold rule) | CI lower bound > 0 — structure beyond adaptivity |
| S2 | `rcasr_bm25` | `ternary_bm25` (argmax posteriors, Kleene gate/state, false → 10⁻⁴) | CI lower bound > 0 — continuous beyond ternary |
| N | `rcasr_bm25` on posteriors permuted across images (seed 20260922) | BM25 rank mixture | expected to fail P1 |

## Also reported regardless of outcome

- **E4**, one factor at a time from full RCASR-BM25 and RCASR-R4, at matched width, with Δ and
  site-bootstrap CI: −gate term, −state term, −continuous (ternary), −abstention (τ = τ_open
  = 0), −opening class (τ_open = τ), fixed k (per-image top-⌈w*⌉ by s instead of the global
  threshold), and pooled vs group CRC. Groups are the label-free family of each image's
  top-scored rule, with a minimum group size of 30 and a pooled fallback.
- **Calibration**: CRC at α ∈ {0.05, 0.10, 0.20} on the calibration fold, cap 3 with
  whole-image deferral. Report held-out empirical risk, the gap to α, mean/median/p95 width and
  the deferral rate. If mean width exceeds 3 at α = 0.10, R2 stays triggered.
- **E2**: every E2 arm (BM25 k=3 mixture, R4 k=3, adaptive BM25, RCASR-BM25, RCASR-R4) is
  judged by one frozen per-pair judge, J3-sym on endpoint `Qwen3.8-27B-FP8` with the frozen
  fact cache. Every arm reads the same pair cache. Reported: GV recall; closed-world precision
  and F1 labelled assumption-dependent; false alarms per image, split into
  contradicts-recorded-compliant / undetermined / unrecorded; misses per image;
  need_review rate; exact-image success; macro by family.
- **E5**: per-site recall for each arm, and leave-one-family-out, where the fit ignores one
  family's positives and that family's held-out recall is reported.
- Harm: recorded violations present in the reference but absent from RCASR, by atom class.
- Cost: model calls, tokens and seconds per image for each pass.

## Public auxiliary (E5b), after the internal endpoints

The final configuration is fitted on all 346 site images, frozen, and only then applied to
the 412 ConstructionSite-10k test images with at least one recorded violation, with facts from
the same 9B server and the same fact prompt as the frozen cache. Arms are BM25 k = 3 and
RCASR-BM25 at matched width. Reported: per-rule detection over scoreable mapped positives at
retrieval level and end-to-end (J3-sym, 27B), with Wilson intervals, native and image-level
derived, split by prior exposure. No precision or F1. R4 is not run on the public set
(an agent loop per image is outside this budget); this is declared.

## Not allowed after the first run

Changing the grids, the prompts, τ classes, budgets, endpoints, bars or the arms listed above;
promoting an exploratory arm to primary; selecting anything on held-out folds.
