# Cross-model judgement gap — paired image-level bootstrap

Oracle judgement (`image_rules`), 2000 resamples, seed 20260716. Reference judge: **Qwen3.6-35B-A3B**. Delta is reference minus model, so a positive delta means the reference is better; a CI straddling zero means the two judges are indistinguishable at 95%.

| judge | n images | reference GV-F1 | judge GV-F1 | delta F1 | 95% CI | significant |
|---|--:|--:|--:|--:|---|---|
| Qwen3.5-9B | 500 | 0.8668 | 0.8569 | +0.0099 | [-0.0043, +0.0250] | no |
| gemma-4-12B | 500 | 0.8668 | 0.8231 | +0.0437 | [+0.0266, +0.0607] | **yes** |

Scaling within a family buys nothing measurable; the model family still matters.
