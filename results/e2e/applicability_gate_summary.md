# Applicability gate — what the gate did

A three-way gate (`applicable` / `not_applicable` / `unknown`) runs ahead of the
symbolic verdict. `not_applicable` filters the candidate outright; `unknown` routes
to `need_review`, a structural abstention rather than a confidence threshold.

`filter_precision` is the share of filtered candidates that gold agrees were not
violated. `filtered_gold_violation` is recall the gate destroyed.

| cell | cand/img | filter | filter prec. | wrongly filtered | abstention | coverage | GV-P | GV-R | GV-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `R4_gated_sym` | 2.238 | 0.0706 | 0.8228 | 14 | 0.1957 | 0.8043 | 0.4246 | 0.4602 | 0.4417 |
| `R4fam_gated_sym` | 6.086 | 0.2935 | 0.9552 | 40 | 0.1669 | 0.8331 | 0.2482 | 0.5384 | 0.3397 |

## Against the no-gate counterpart

| cell | GV-P | GV-R | GV-F1 | baseline | GV-P | GV-R | GV-F1 |
|---|---:|---:|---:|---|---:|---:|---:|
| `R4_gated_sym` | 0.4246 | 0.4602 | 0.4417 | `R4_decoupled_sym` | 0.42 | 0.5847 | 0.4888 |
| `R4fam_gated_sym` | 0.2482 | 0.5384 | 0.3397 | `R4fam_decoupled_sym` | 0.2083 | 0.7164 | 0.3228 |

### `R4_gated_sym` symbolic routes

| route | n |
|---|---:|
| formula_true | 749 |
| gate_unknown | 219 |
| gate_not_applicable | 79 |
| formula_false | 40 |
| unknown_needs_review | 31 |
| not_visually_evaluable | 1 |

### `R4fam_gated_sym` symbolic routes

| route | n |
|---|---:|
| formula_true | 1499 |
| gate_not_applicable | 893 |
| gate_unknown | 508 |
| formula_false | 86 |
| unknown_needs_review | 48 |
| no_subject_default | 8 |
| not_visually_evaluable | 1 |
