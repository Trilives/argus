# Low-width E2 — pre-specification

**Written:** 2026-09-22, before any low-width selection was scored or any low-width
end-to-end number was seen. **Do not edit after the first run.** Outcomes go in a separate
record. Authorisation: user decision 2026-09-22 (`../history/TASK_LOG.md` §2, "Low-width E2
arm authorised"). Risk register entry: R10 (`../Task.md` §C).

This file is also the **stabilisation point** required by `../Task.md` §E6: from the moment it
exists, nothing in the scorer, its fitted weights, the atom questions, the abstention rule or
the selection rule moves. A later change needs a new freeze and a new log §2 entry.

## Question

The width-3 E2 (`../../Records/2026-09-22/rcasr_e2e_public.md`) showed that RCASR-BM25 raises
grounded-violation recall (GV-R 0.434 → 0.625) but also false alarms per image (1.28 → 1.56)
at the same mean width as BM25 top-3. RQ2 is therefore "mixed". This experiment asks whether
RCASR-BM25 **at a smaller budget** keeps the recall of BM25 at width 3 while cutting the
judged pairs and the false alarms per image. If it does, RQ2 becomes a review-workload
claim. If it does not, the null is reported as a null and the width is never re-picked.

## Status of the data

All internal images are development-exposed. Everything below is retrospective,
site-grouped and cross-fitted, not an independent test. The judge verdicts come from the one
frozen per-pair judge of the width-3 E2 (`results/2026-09-22_rcasr_judge/`), so the judge is
unchanged, its two serving stacks and 11 parse errors carry over, and no verdict is re-judged.

## The one target width

**Target mean width $w_{low} = 2.0$.** Fixing rule, applied to the post-hoc F5 diagnostic
(`../manuscript/figures/f5_data.json`, computed 2026-09-22 before this file): the smallest
width on the F5 grid $\{1, 1.5, 2, 2.5\}$ at which held-out RCASR-BM25 **retrieval** recall
exceeds the BM25 top-3 retrieval recall of 0.515. At width 1.5 RCASR-BM25 reaches 0.460
(below); at width 2 it reaches 0.554 (above). Linear interpolation puts the crossing near
1.8; the grid point 2.0 is used, not the interpolation, so the arm runs at a one-third budget
cut with no fitted quantity re-chosen. No other width is an arm.

## Arms

| id | selection | judged pairs |
| --- | --- | --- |
| **`rcasr_bm25_w2`** (the arm) | fold-fitted RCASR-BM25 scores from the v2 freeze, unchanged $(\theta,\tau,\tau_{open})$ per fold; one global threshold per fold set on the **fitting folds** at the $(n_{fit}\cdot 2.0)$-th largest score, applied to the held-out fold | read from the frozen cache |
| `bm25_w3` (primary reference) | the width-3 BM25 rank mixture of the width-3 E2 (`bm25_matched`), unchanged | cached |
| `rcasr_bm25_w3` (same method, full budget) | the width-3 RCASR-BM25 arm of the width-3 E2, unchanged | cached |
| `bm25_w2_matched` (supporting) | BM25 rank mixture at the arm's realised total width (`matched_rank_sets`) | cached (subset of top-3) |

Because the fitted parameters are unchanged and the threshold rises monotonically with a
smaller budget, every `rcasr_bm25_w2` set is a subset of the corresponding width-3 set, so
every pair already has a frozen verdict. The script asserts the subset relation. If any pair
is nevertheless missing from the cache, it is judged by the **same frozen judge, unchanged**
(`judge_rcasr.judge_pair`, guided JSON, `Qwen3.8-27B-FP8`), appended to a cache inside the new
freeze, and the count is reported. No re-judging of cached pairs.

The §3.4 `no_subject_default` question is **not** folded in: the judge is the frozen one.

## Endpoints (paired, site-unit bootstrap, 2,000 replicates, seed 20260922)

Both against `bm25_w3` on the same 346 images. Site units are the 48 confirmed site units of
the v2 freeze (near-duplicate clusters fused).

| id | statistic | GO if |
| --- | --- | --- |
| **W1 recall non-inferiority** | $\Delta$GV-R $=$ GV-R(`rcasr_bm25_w2`) $-$ GV-R(`bm25_w3`) | 95% CI lower bound $> -0.03$ (margin = the project's recall bar) |
| **W2 fewer false alarms** | $\Delta$FA/img $=$ FA/img(`rcasr_bm25_w2`) $-$ FA/img(`bm25_w3`) | 95% CI upper bound $< 0$ |

The review-workload claim is made only if **both** pass. One passing is reported as such and
does not support the claim. Point estimates without their intervals are never quoted.

## Also reported regardless of outcome

- Mean/median/p95 width of the arm (= judged pairs per image), and the share of images with
  an empty set.
- Tokens per image: extractor (from the evidence freeze, fixed per image) and judge (from the
  cached judge rows where usage is recorded; else pairs per image × the width-3 mean per pair,
  labelled as such).
- GV-P and GV-F1 (closed-world, assumption-dependent), misses per image, `need_review`
  rate, exact-image success, macro recall by family, false alarms split into
  contradicts-recorded-compliant / undetermined / unrecorded.
- The same statistics for `rcasr_bm25_w3` and `bm25_w2_matched`, with paired CIs against
  `bm25_w3` for context.
- **Post-hoc diagnostic, no gate, no claim:** the arm's GV-R and FA/img at the F5 grid
  widths $\{1, 1.5, 2, 2.5, 3\}$, all from the cache, to show the shape of the trade-off.
  This curve never replaces the pre-specified point.

## Not allowed after the first run

Changing the target width, the endpoints, the margin, the reference arm, the bootstrap seed
or the judge; adding a width; promoting the diagnostic curve to a result; refitting any
parameter; folding in the §3.4 default.
