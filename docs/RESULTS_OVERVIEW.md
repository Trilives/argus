# Results overview

[Result files](../results/README.md) · [Analysis plans](research/README.md) ·
[Analysis timing](RESEARCH_DISCLOSURES.md#analysis-timing)

The additional images are unlabelled research inputs and are not scored as a new evaluation set.

| Analysis | Reported result and limit |
|---|---|
| Primary internal | 346 images, 48 sites, 42 provisions; site-grouped cross-fitting. Matched-width-3 recorded-violation recall 0.515 (BM25) → 0.709. RCASR beat an unstructured threshold and ternary screen and matched a multi-turn agent. |
| Later width-2 arm | Kept BM25's width-3 recall; false alarms per image 1.28 → 1.00. Specified after primary results and width-2 retrieval diagnostics, before width-2 end-to-end scoring. |
| ConstructionSite-10k | Frozen before public scoring. Harness detection 0.08 → 0.80, PPE 0.43 → 0.67; performance fell for the excavator-radius provision. Gated inputs are not bundled. |
| Two-stage prior | SigLIP 2 then cross-encoder: matched-width recall 0.509 → 0.640 (+0.132, 95% site CI [+0.085, +0.171]). Fixed top-3 was no better than BM25; RCASR on this prior remained below RCASR on BM25. |
| Single-model chain | The full-precision 9B chain passed the stated width-2 gates. The robot FP8 checkpoint did not establish recall non-inferiority; deployment needs its own labelled validation. |
| Conformal risk control | Met the miss-risk target with about ten candidates per image at α = 0.10. |
| Patrol robot | Runtime parity and stored-round operating proxies; no independent field-accuracy claim. |
| Revision H1–H8 | Post-hoc statistics, applicability, atom errors, data description and vocabulary analyses over frozen outputs. |

The ARGus baseline measured **GV-F1 0.867** with oracle rules and **0.514** for the best
complete system. Retrieval dominated errors, motivating RCASR. The bundled label-only scorer
reproduces the latter value; see [the reproduction guide](../reproduce/REPRODUCE.md).
