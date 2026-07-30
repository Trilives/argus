# Error composition, R4 x image_rules (rebound frozen gold)

- gold violations 691: TP 430, FN 261, FP 552
- recall 0.6223, precision 0.4379 (must match the published 0.622 / 0.438)
- FN: retrieval miss 221 (84.7%) = sibling-in-candidates 143 + no-same-category 78; judgment 40 = judged-compliant 26 + abstained 14
- FN by category: {'EDG': 98, 'CIV': 53, 'OPN': 66, 'BHV': 44}
- retrieval-miss FN by category: {'EDG': 91, 'CIV': 49, 'OPN': 57, 'BHV': 24}
- FP: same-category-as-a-gold-violation 216, other 336
