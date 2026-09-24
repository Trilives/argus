# RCASR scorer — module proposal (awaiting approval)

**Status:** proposal only, 2026-09-22. Nothing below has been implemented or run. Required
by `docs/ARCHITECTURE.md` "Adding a module". Once it is approved, a separate prespec with
fixed gates is written before any inference, as for `SUBJECT_SCREEN_PRESPEC.md`.

## What it closes (Todo item 1)

| Todo | Proposal |
| --- | --- |
| (a) continuous $s_\theta(x,r)$ | per-atom posteriors from token log-probabilities; the compiled gate/state formulas are evaluated probabilistically, not in Kleene logic |
| (b) state/visibility observations | a second rule-agnostic pass over the visually observable state, hazard, visibility and violation atoms in `typed_evidence_v1.json` (~75, `external_only = false`) |
| (c) abstention calibration + opening/shaft/edge class | abstention threshold τ fitted on training folds only, with a separate τ for the six opening/shaft/edge atoms; unknown never excludes |
| (d) E4 ablation | one-at-a-time arms, listed below |
| (e) hierarchical routing | not built. Reported as a negative from the existing measurement (168/183 atoms serve one rule) |

## Score

For image $x$ and rule $r$, with retriever prior $\pi(x,r)$ (the rank-normalised BM25
score, or R4 membership/rank):

$$s_\theta(x,r)=\theta_0\,\mathrm{logit}\,\pi(x,r)+\theta_1\log P_{\text{gate}}(x,r)+\theta_2\log P_{\text{state}}(x,r)$$

- $P(a\mid x)$ for each atom is the renormalised probability of `yes` over {`yes`,`no`} at the
  answer token. If the model's `unclear` mass exceeds τ, the atom is unknown, and an unknown
  atom contributes $P=1$ to the gate, so it cannot withhold.
- The gate uses AND → product, OR → noisy-or and NOT → $1-p$, over the compiled subject gate.
  $P_{\text{state}}$ is the same evaluation over the rule's state tests (violation direction).
- There are three weights plus τ. They are fitted by site-grouped cross-fitting to maximise
  recorded-violation recall at a fixed width budget. Recorded violations are the only
  positives, and unlabelled pairs are never used as negatives.
- Selection: CRC on the held-out calibration fold (`set_selection.select`), primary α = 0.10,
  cap 3 with deferral to `need_review` (existing `capped_selection`). The headline stays
  **recall at matched mean width**.

## Evaluation (after site sign-off)

- Folds: confirmed sites plus near-duplicate clusters, K = 5, site-grouped. The ~31% of images
  without a site stay out of the site analyses and appear only in an all-500 resubstitution
  appendix. Everything remains retrospective and development-exposed.
- Arms: BM25 k∈{2,3,4}, R4 k∈{2,3}, unstructured CRC, the ternary subject screen, and full
  RCASR over BM25 and over R4.
- E4, one factor at a time: −state term, −continuous (ternary), −calibrated τ (fixed 0.5),
  −opening-class τ, fixed-k instead of CRC, pooled instead of group calibration.
- E2 end to end: one frozen per-pair judge (J3-sym) over the union of pairs any arm selects,
  so every arm reads the same judgements.
- E5: per-site metrics, leave-one-site-out and leave-one-family-out, with cluster-bootstrap
  CIs.

## New files

| File | Owns | Test |
| --- | --- | --- |
| `src/atom_posterior.py` | prompts, log-prob parsing and posteriors for the two rule-agnostic passes | `tests/test_atom_posterior.py` |
| `src/rcasr.py` | probabilistic formula evaluation, the score, τ abstention, weight fitting | `tests/test_rcasr.py` |
| `src/rcasr_experiment.py` | freeze / infer / evaluate stages, arms, ablations | covered by `tests/test_rcasr.py` |
| `src/site_partition.py` | confirmed sites + near-duplicate clusters → grouped folds | `tests/test_site_partition.py` |
| `experiments/retrieval/eval_rcasr.py` | thin CLI (`--prepare/--infer/--evaluate`) | — |
| `data/rules/proposed/rcasr_v1.json` | frozen config: model, prompts, folds seed, α, cap | — |

Every existing `src/` file stays byte-identical, because all of them are pinned by freezes.
The new code only imports them.
