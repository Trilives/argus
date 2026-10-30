# Two-stage retrieval prior (Todo 3-B) — pre-specification

**Written:** 2026-09-27, before the two-stage retriever below was run on any internal image and
before any RCASR score on its prior was computed. **Do not edit after the first run.** Outcomes
go in a separate record. Authorisation: `../Todo.md` item 3-B (future work, does not block
submission). Frozen configuration: `data/rules/proposed/rcasr_v1_tsr.json`.

## Question

The Discussion (§5, "Relation to fixed-cardinality retrieval-augmented screening") says: *"A
bi-stage retriever could supply the prior π that RCASR re-weights, in the same way BM25 and the
agent retriever do here."* This experiment tests that sentence. It asks one thing: dropped in as
a third prior, unchanged in every other respect, does RCASR raise the recall of a vector-retrieval
+ cross-encoder-rerank retriever at matched width?

## Status of the data

All internal images are development-exposed. Everything below is retrospective, site-grouped
and cross-fitted, not an independent test. Exposure of the components: the cross-encoder, used
alone over all 42 provisions, was scored on all 500 gold images on 2026-07-28 (hit@3 0.730);
SigLIP-2-224 alone was scored on the pre-refreeze 200-image gold (hit@3 0.595, a voided gold).
The cascade of the two has never been run, and no choice below was made from a new
label-reading computation.

## The retriever (TSR)

- **Stage 1, vector retrieval.** SigLIP-2 `google/siglip2-base-patch16-224` (the project's default
  variant, `src/config.py` `SIGLIP_VARIANT`), code `src/retrieval/siglip.py` unchanged. The query
  is the image (`images.load_image`); each provision is embedded from its `visual_retrieval_text`
  in `data/rule_assets/rule_index.json`; cosine similarity; the top **M = 10** go to stage 2.
- **Stage 2, rerank.** Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, code
  `src/retrieval/cross_encoder.py` unchanged. The query is the image's generic scene description
  from the frozen fact cache (`results/retrieval/gold_facts_generic_en.json`, via `facts_query`,
  the same query the BM25 prior uses); the document is the provision's `evidence_chain_text`. The
  M candidates are reordered by cross-encoder score, ties by stage-1 rank.
- **Output:** the reranked list of length M. Provisions not in stage 1 are not retrieved.
- **Why M = 10:** about a quarter of the 42-provision library, above the width-3 budget with room
  for re-ordering, and a conventional cascade depth. It is fixed now; no other M is run.
- **Prior map.** $\pi=(M+1-\text{rank})/(M+1)$ for ranks 1..M and $0.5/(M+1)$ for unretrieved
  provisions. This is the manuscript's R4 map written for a list of length $L$: $L=4$ gives R4's
  $(5-\text{rank})/5$ and 0.1 exactly, so no new mapping parameter is introduced. Ties among
  unretrieved provisions are broken by the existing hashed tie-break.
- Weights are read offline from the local Hugging Face cache and pinned by revision and file
  SHA-256 at `--prepare`. Inference is fp32, one image at a time, on the local RTX 4090. The
  retriever reads no label.

## Method

Unchanged from the RCASR v2 freeze (`results/2026-09-22_rcasr_v2`, `RCASR_PRESPEC.md`): the same
346 site images, 48 site units, K = 5 folds and seed, calibration fold
$(f+1)\bmod 5$, the frozen 9B atom evidence (`results/2026-09-22_rcasr_evidence`), the grid, the
fitting objective, the tie-break, the bootstrap (2,000 site-unit replicates, seed 20260922). Every
method function is imported from `src/rcasr_experiment.py` unchanged; only the prior differs.
**Budget $w^\ast = 3.0$**, the BM25 rule: the TSR top-3 always has three members.

## Arms (cross-fitted held-out, 346 site images)

The nine variants of `rcasr_experiment.variants` on the TSR prior: `rcasr_tsr`, `adaptive_tsr`,
`ternary_tsr`, `no_gate_tsr`, `no_state_tsr`, `no_abstention_tsr`, `no_open_class_tsr`,
`fixed_k_tsr`, `null_tsr`. Each is compared with the TSR rank mixture at its realised total width
(`matched_rank_sets`).

## Gates

| id | arm | reference | GO if |
| --- | --- | --- | --- |
| **T1** (primary) | `rcasr_tsr` | TSR rank mixture, matched width | Δ recall ≥ +0.03 and 95% site CI lower bound > 0 |
| T2 | `rcasr_tsr` | `adaptive_tsr` | CI lower bound > 0 (structure beyond adaptivity) |
| TN | `null_tsr` | TSR rank mixture, matched width | expected to fail T1's bar |

**Wording, fixed now.** T1 passes: the Discussion sentence may state that a bi-stage retriever
supplied the prior and RCASR raised its recall at matched width, with Δ and CI, labelled
retrospective. If T1 passes and T2 fails, the gain is described as reallocation across images, as
for R4. T1 fails: the sentence reports the null with Δ and CI; M, the models, the text fields and
the map are not re-picked. Point estimates are never quoted without their intervals.

## Also reported regardless of outcome (no gate)

- **Context at width 3, same images, paired site CIs:** TSR top-3 vs BM25 top-3 (the raw priors);
  `rcasr_tsr` vs the frozen v2 held-out `rcasr_bm25` (both fitted at $w^\ast=3$; realised widths
  reported); TSR top-3 vs stage-1-only top-3 (what the rerank adds). R4 top-3 recall is shown at
  its own width (≈2.24) as context only, not as a matched comparison.
- One-factor ablations (the variants above) with Δ and CI against their matched references.
- CRC at α ∈ {0.05, 0.10, 0.20}, pooled and group, as in the v2 freeze.
- Leave-one-family-out, per-site recall, harm (recorded violations the matched reference kept
  and `rcasr_tsr` dropped).
- Cost: seconds per image for each stage on the 4090; parameter counts of both models.

## Not run

- **E2 (end-to-end judge).** The claim under test concerns the retrieval prior; a judge run
  would add new pairs outside the frozen judge cache. If T1 passes, an E2 needs its own
  pre-specification.
- The public auxiliary set, the width-2 operating point, and any width other than 3.

## Not allowed after the first run

Changing M, either model, the text fields, the query, the prior map, the budget, the gates or
their bars; adding an arm or a width; promoting a context comparison to a gate; selecting
anything on held-out folds.
