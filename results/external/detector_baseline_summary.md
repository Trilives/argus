# Dedicated-detector (YOLO hard-hat) baseline — BHV/PPE

Helmet-compliance head-to-head on **R-BHV-002** over all 500 gold images. Detector = keremberke/yolov8m-hard-hat-detection {Hardhat, NO-Hardhat}; a NO-Hardhat box (conf ≥ τ) ⇒ predicted helmet violation. Same grounded-violation TP definition as the main table, restricted to the helmet rule.

Gold: 101 helmet-violation images, 136 rule-applicable, 500 total.

## Detector confidence-threshold sweep

| τ | fired | TP | FP | FN | P | R | F1 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0.10 ⭐ | 44 | 29 | 15 | 72 | 0.659 | 0.287 | 0.4 |
| 0.15 | 33 | 24 | 9 | 77 | 0.727 | 0.238 | 0.358 |
| 0.20 | 24 | 19 | 5 | 82 | 0.792 | 0.188 | 0.304 |
| 0.25 | 22 | 18 | 4 | 83 | 0.818 | 0.178 | 0.293 |
| 0.30 | 19 | 15 | 4 | 86 | 0.789 | 0.149 | 0.25 |
| 0.35 | 15 | 12 | 3 | 89 | 0.8 | 0.119 | 0.207 |
| 0.40 | 11 | 9 | 2 | 92 | 0.818 | 0.089 | 0.161 |
| 0.50 | 5 | 4 | 1 | 97 | 0.8 | 0.04 | 0.075 |
| 0.60 | 2 | 2 | 0 | 99 | 1.0 | 0.02 | 0.039 |

## Head-to-head on the helmet rule (R-BHV-002)

| system | P | R | F1 |
|---|--:|--:|--:|
| YOLO hard-hat (best τ=0.10) | 0.659 | 0.287 | 0.4 |
| Ours — R4×J2 (grounded pipeline) | 0.925 | 0.614 | 0.738 |
| YOLO hard-hat, applicable-only (best τ=0.10) | 0.967 | 0.287 | 0.443 |

## Why the detector misses (mechanism, not silent failure)

On the 101 helmet-violation images: **0 have no detection at all**, **70 show only Hardhat boxes** (the detector sees helmeted heads but the violation is improper-wearing/attire/chin-strap — outside its binary class scheme — or an extra bare head it missed), and 31 show a NO-Hardhat box. The specialist is **high-precision, low-recall**: when it fires it is almost always right, but it cannot recall the full rule.

> **Caveat.** This is an *off-the-shelf* detector (not trained on these site images). An in-domain-trained detector would recover helmet-presence recall at PPE-box annotation cost, but the coverage limit below is structural and holds regardless.

## Coverage: what a specialist can even address

- Total gold violation pairs: **691**
- By category: {'OPN': 243, 'EDG': 250, 'BHV': 111, 'CIV': 87}
- A helmet detector can address the helmet rule only: **101/691 = 14.6%** of the violation workload.
- A helmet+harness PPE suite: **15.1%**.
- The grounded pipeline addresses all 42 rules / **100%** of the taxonomy.
