# Split leakage — tuned-on vs held-out re-score

The frozen gold is a seeded draw from the full 1578-image pool taken independently of
`data/splits/`, so it straddles every split. This re-scores the *cached* end-to-end
predictions on disjoint strata — no re-inference. `tuned_dev_subset` = gold images that
sit in `development_subset_images.txt` (prompt/threshold tuning); `held_out` = the rest.
Delta CIs come from independent within-stratum bootstrap (2000 reps, seed 20260728).

## Gold composition per stratum

| stratum | images | gold violations | positive-image rate | violations/image | applicable/image |
|---|---:|---:|---:|---:|---:|
| all_500 | 500 | 691 | 0.944 | 1.382 | 1.76 |
| tuned_dev_subset | 105 | 130 | 0.8857 | 1.2381 | 1.7048 |
| held_out | 395 | 561 | 0.9595 | 1.4203 | 1.7747 |
| held_out_clean | 381 | 540 | 0.9606 | 1.4173 | 1.7743 |
| train_images | 355 | 499 | 0.9549 | 1.4056 | 1.7662 |
| val_images | 72 | 96 | 0.9167 | 1.3333 | 1.7222 |
| realistic_test_images | 73 | 96 | 0.9178 | 1.3151 | 1.7671 |

## Per-cell scores by stratum

| cell | stratum | n | GV-P | GV-R | GV-F1 | GV-F1 95% CI |
|---|---|---:|---:|---:|---:|---|
| R4_image_rules | tuned_dev_subset | 105 | 0.398 | 0.6154 | 0.4834 | [0.4211, 0.5418] |
| R4_image_rules | held_out | 395 | 0.4481 | 0.6239 | 0.5216 | [0.4915, 0.5522] |
| R4_image_rules | held_out_clean | 381 | 0.452 | 0.6278 | 0.5256 | [0.4956, 0.5589] |
| R4_image_rules | train_images | 355 | 0.4326 | 0.6172 | 0.5087 | [0.4781, 0.5392] |
| R4_image_rules | val_images | 72 | 0.4406 | 0.6562 | 0.5272 | [0.4475, 0.6027] |
| R4_image_rules | realistic_test_images | 73 | 0.4646 | 0.6146 | 0.5291 | [0.4434, 0.6106] |
| R4_decoupled_sym | tuned_dev_subset | 105 | 0.366 | 0.5462 | 0.4383 | [0.372, 0.5045] |
| R4_decoupled_sym | held_out | 395 | 0.4336 | 0.5936 | 0.5011 | [0.4699, 0.5322] |
| R4_decoupled_sym | held_out_clean | 381 | 0.4381 | 0.5963 | 0.5051 | [0.4734, 0.5374] |
| R4_decoupled_sym | train_images | 355 | 0.4171 | 0.5852 | 0.4871 | [0.4543, 0.5189] |
| R4_decoupled_sym | val_images | 72 | 0.4307 | 0.6146 | 0.5064 | [0.4327, 0.5787] |
| R4_decoupled_sym | realistic_test_images | 73 | 0.424 | 0.5521 | 0.4796 | [0.3962, 0.5577] |
| R3_image_rules | tuned_dev_subset | 105 | 0.3073 | 0.5154 | 0.3851 | [0.3275, 0.4432] |
| R3_image_rules | held_out | 395 | 0.3786 | 0.5615 | 0.4523 | [0.4207, 0.4856] |
| R3_image_rules | held_out_clean | 381 | 0.3812 | 0.5648 | 0.4552 | [0.4239, 0.4876] |
| R3_image_rules | train_images | 355 | 0.3606 | 0.5471 | 0.4347 | [0.4032, 0.466] |
| R3_image_rules | val_images | 72 | 0.4138 | 0.625 | 0.4979 | [0.4167, 0.5854] |
| R3_image_rules | realistic_test_images | 73 | 0.3311 | 0.5104 | 0.4016 | [0.3293, 0.4732] |
| BM25_image_rules | tuned_dev_subset | 105 | 0.2634 | 0.4923 | 0.3432 | [0.2919, 0.3947] |
| BM25_image_rules | held_out | 395 | 0.2747 | 0.4563 | 0.3429 | [0.316, 0.3671] |
| BM25_image_rules | held_out_clean | 381 | 0.2746 | 0.4556 | 0.3426 | [0.3167, 0.3679] |
| BM25_image_rules | train_images | 355 | 0.2758 | 0.4609 | 0.3451 | [0.3171, 0.3703] |
| BM25_image_rules | val_images | 72 | 0.264 | 0.4896 | 0.3431 | [0.2891, 0.3929] |
| BM25_image_rules | realistic_test_images | 73 | 0.2638 | 0.4479 | 0.332 | [0.2646, 0.3983] |
| R1_image_rules | tuned_dev_subset | 105 | 0.1927 | 0.3231 | 0.2414 | [0.1886, 0.2978] |
| R1_image_rules | held_out | 395 | 0.2074 | 0.3191 | 0.2514 | [0.2236, 0.28] |
| R1_image_rules | held_out_clean | 381 | 0.2104 | 0.3222 | 0.2546 | [0.2267, 0.2851] |
| R1_image_rules | train_images | 355 | 0.2057 | 0.3186 | 0.25 | [0.2212, 0.2809] |
| R1_image_rules | val_images | 72 | 0.1867 | 0.3229 | 0.2366 | [0.1692, 0.3071] |
| R1_image_rules | realistic_test_images | 73 | 0.2183 | 0.3229 | 0.2605 | [0.1909, 0.332] |

## Tuned-minus-held-out delta

Positive = the tuned subset scores higher, i.e. the all-500 headline is inflated by tuning.
`held_out_clean` additionally drops held-out images that near-duplicate a tuning image
(`results/data_audit/near_duplicates.json`), so it is the strictest never-seen subset.

| cell | all-500 GV-F1 | held-out GV-F1 | ΔGV-F1 | 95% CI | sig. | ΔGV-P | ΔGV-R | clean GV-F1 | Δ vs clean | 95% CI | sig. |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|
| R4_image_rules | 0.514 | 0.5216 | -0.0382 | [-0.1063, 0.0278] | no | -0.0501 | -0.0085 | 0.5256 | -0.0422 | [-0.1091, 0.023] | no |
| R4_decoupled_sym | 0.4888 | 0.5011 | -0.0628 | [-0.1352, 0.0093] | no | -0.0676 | -0.0474 | 0.5051 | -0.0668 | [-0.141, 0.0053] | no |
| R3_image_rules | 0.4388 | 0.4523 | -0.0672 | [-0.129, -0.0029] | yes | -0.0713 | -0.0461 | 0.4552 | -0.0701 | [-0.137, -0.0033] | yes |
| BM25_image_rules | 0.343 | 0.3429 | 0.0003 | [-0.0563, 0.0583] | no | -0.0113 | 0.036 | 0.3426 | 0.0006 | [-0.0597, 0.0575] | no |
| R1_image_rules | 0.2494 | 0.2514 | -0.01 | [-0.0688, 0.0544] | no | -0.0147 | 0.004 | 0.2546 | -0.0132 | [-0.0728, 0.0483] | no |
