# Revision post-hoc analyses (*Buildings* 4625629, major revision)

Written 2026-10-24, before any analysis below was run. Every analysis here is **post hoc**: all
primary results of the paper were seen before this plan. They answer reviewer requests (R1.5b,
R1.7, R1.8, R1.9, R3.1–R3.4, R2.C7, R1.4) and change no reported number. No model is called;
everything is computed from frozen outputs. The reviewer-to-analysis map is in
`../Journal_response/Need/Todo.md` §4–§5 (H1–H8 here = A1–A8 there).

Entry points (new files only; nothing under `src/` changes):
`experiments/retrieval/eval_revision_posthoc.py --prepare / --run` (H1, H2, H4, H5, H6) and
`experiments/retrieval/eval_revision_cases.py --run` (H3, H7, H8), both writing to
`results/2026-10-24_revision_posthoc/`. `--prepare` writes `input_snapshot.json` over the plan, both
scripts, the frozen inputs below and their own snapshots; `--run` verifies it first.
Record: `docs/Records/2026-10-24/revision_posthoc.md`.

Frozen inputs: the v2 analysis freeze (`results/2026-09-22_rcasr_v2`), the width-2 freeze
(`results/2026-09-22_rcasr_lowwidth`), the judge cache (`results/2026-09-22_rcasr_judge`), the public
freeze (`results/2026-09-22_rcasr_public`), the robot round reports
(`deploy/Output/dog-20260923/reports/dog0910{,r}-{28,29}.json`,
`deploy/Output/dog-20260927/reports/dog0910f-{28,29}.json`), `data/rules/rules_en.json`, the gold, and
`data/splits/site_keys_candidate.json`. Bootstrap seed and repeats are the frozen
`rcasr_v1.json` values unless stated.

## H1 Provisions that cannot be judged from a photograph (R3.4)

N = the provisions whose `visual_screening_rule` is `not_available_for_single_image` in
`rules_en.json` (expected: BHV-006, BHV-011, CIV-006, CIV-008, CIV-009).

- **H1a, internal, removed after selection.** For RCASR-BM25 at width 2 and 3 and BM25 at width 3
  (`lowwidth_rows.jsonl`), drop N from each set and recompute the end-to-end metrics with the frozen
  verdicts. Report each metric and its change.
- **H1b, internal, removed before thresholding.** Fold-fitted parameters unchanged; scores of N set to
  −∞; the per-fold threshold re-set on the fitting folds at mean width 2 and 3 over the other 37
  provisions. Retrieval recall of recorded violations for both widths. End-to-end metrics only if every
  selected pair has a frozen verdict; otherwise report the number of unjudged pairs and stop at
  retrieval.
- **H1c, robot round 28 + 29.** For the three stored runs (frozen threshold; site-recalibrated
  threshold; FP8 on the robot's GPU), remove N's verdicts from each frame and recompute the frame
  decision with the runtime rule: `violation` if any verdict is `non_compliant`; else `need_review` if
  any is `need_review` or missing; else `clear` if any pair was judged; else `clear_empty`. The script
  first asserts that this rule reproduces every stored decision. Report decision shares, judged pairs
  per frame and the share of judged pairs routed to review, before and after. The freed width is not
  refilled (that needs new judge calls).

## H2 Applicable provisions that are not violated (R3.1)

On the 346 site images, for six arms (BM25 width 3 matched mixture, RCASR-BM25 width 3, BM25 width 2
matched mixture, RCASR-BM25 width 2, R4 top-3, RCASR-R4), the share of recorded `compliant` pairs,
recorded `undetermined` pairs and all recorded pairs (any status) that the set contains. Paired
site-unit bootstrap CIs for RCASR-BM25 width 3 − BM25 width 3 and RCASR-BM25 width 2 − BM25 width 3.
Descriptive: recorded compliant pairs are not a random sample of applicable pairs.

## H3 How atom errors propagate (R3.3, R1.7)

- **H3a.** Fitted (θ, τ, τ_open) per fold for RCASR-BM25 and RCASR-R4. Over the 346 images, the share of
  atom posteriors that are *known* under each fold's (τ, τ_open), separately for the six opening-class
  atoms and the rest, and the share of known atoms whose confidence c = (1 − u)·max(q, 1 − q) is
  below 0.8.
- **H3b.** For each of the 29 lost pairs (`harm.bm25.lost_pairs` of the v2 freeze), under its held-out
  fold's parameters: prior logit, gate term θ₁·log P_G, state term θ₂·log P_V, score − threshold, the
  rank of the provision in the image. The **pivotal atom** is the atom in the provision's gate or
  screening formula whose literal, set to 1 (unknown), raises the score most; record its q, u, c,
  class, and whether that change alone would select the pair. Exclusive categories, fixed here:
  (i) *outranked*: the evidence terms sum to more than −0.1, so the prior alone left the pair out;
  (ii) *confident contradiction*: otherwise, pivotal atom c ≥ 0.8;
  (iii) *uncertain atom read as evidence*: otherwise, pivotal atom c < 0.8.
  Counts by category, family and opening class.
- **H3c.** Public rule_4 (24 scoreable positives): the same decomposition for each mapped provision
  under the frozen public configuration and threshold; counts by category.
- **H3d, cases.** (1) The F6 panel (b) pair: the F6 image whose recorded R-OPN-001 violation is in BM25
  width 3 but not in RCASR-BM25 width 2, decomposed at the width-2 threshold of its fold. (2) The first
  public rule_4 pair, by image id, that BM25 retrieved and RCASR did not. Full atom tables for both.

## H4 Further CRC operating points (R3.2, R1.9)

`calibration_arm` unchanged, BM25 and R4 priors, pooled and grouped: α ∈ {0.30, 0.40} at cap 3, and
cap 5 at α ∈ {0.10, 0.20, 0.30}. Held-out risk, gap, mean and p95 width, deferral, recall.

## H5 Statistics (R1.8)

- **H5a.** Re-bootstrap P1, P2, S1, S2 with the frozen seed and repeats; the 95% intervals must equal the
  reported ones. Then Bonferroni 98.75% intervals from the same draws (the first Holm step; if all four
  exclude zero, Holm rejects all four).
- **H5b.** Monte Carlo error: P1 and the width-2 recall difference (W1) re-run with seeds +1…+20 at the
  frozen repeats; SD of each bound; and one run at 20,000 replicates.
- **H5c.** Per-site exact two-sided sign test (ties excluded), RCASR-BM25 and RCASR-R4 against their
  matched references.
- **H5d.** Leave-one-family-out, BM25 prior: site-unit bootstrap CI of held-out family recall minus the
  matched reference, per family.
- **H5e.** End-to-end recall by family, RCASR-BM25 width 2 and 3 against BM25 width 3, with site-unit
  bootstrap CIs.
- **H5f.** The 0.03 non-inferiority margin expressed as recorded violations on the 346 images and as a
  share of BM25's width-3 grounded-violation recall.

## H6 The 154 images without a site (R1.5b)

The configuration fitted on all 346 site images (`results/2026-09-22_rcasr_public/final_config.json`,
frozen before the public run) is applied to the 154 images; thresholds are set on the 346 images'
scores at mean width 3 and 2, so nothing about the 154 is used. Reference: the BM25 matched rank
mixture at the same total width on the 154. Retrieval recall of recorded violations; bootstrap over
near-duplicate components (sites are unknown). Also: family composition of the 154 against the 346,
and the number of the 154 that share a near-duplicate component with a site image. Not
site-independent; development-exposed like all internal images.

## H7 What the photographs contain (R2.C7, R1.5d)

From the gold and site keys only, no model output: per-family and per-provision counts of recorded
statuses on the 346 and the 500; images per site (min, median, max); recorded violations per image;
images with no recorded violation. A coarse project type from the type word of the site name, by this
fixed rule, first match wins: residential (花园, 住宅, 公寓, 家园, 苑); office or commercial (大厦, 广场,
商业, 写字楼); industrial or technology park (科技园, 产业园, 工业园, 园区); cultural or public venue (馆,
学校, 医院, 中心); otherwise unspecified. Site names are never written out.

## H8 The atom vocabulary (R1.4)

From the compiled schema and the two question sets: atoms per pass; for each provision, its gate and
screening atoms; provisions per atom; the opening-class six. Written as a table for the supplement.

## What may be claimed

Descriptive, post-hoc statements only. No reported number is replaced. H1b may be stated as what
removing the non-visual provisions does at retrieval level and as a deployment recommendation, not as a
method improvement. H6 is not site-independent. H2 bounds nothing about unrecorded applicable pairs.
