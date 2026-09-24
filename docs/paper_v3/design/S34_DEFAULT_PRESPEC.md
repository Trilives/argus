# §3.4 no-subject default — pre-specification

**Written:** 2026-09-23. No arm-B verdict has been computed and no outcome has been seen.
**Do not edit after the first run.** Outcomes go in a separate record. Authorisation: the
user's §3.4 decision of 2026-09-23 was "optimize it", given in answer to the open item in
`RCASR_RUNTIME_PARITY.md` §3 and `../Task.md` §3.4.

## Question

J3-sym (`src/symbolic_judgement.py:symbolic_verdict`) sends a pair to the rule's
`no_subject_default` whenever the formula is unknown *because its subject gate is unknown*.
The gate is unknown when the subject checkpoint is `not_visible`, `need_review` or missing.
That covers an empty evidence object. The default is `compliant` on all 42 rules.

A verdict should not assert compliance for a provision whose subject was never assessed.
The question is whether abstaining in that branch changes any reported number, and what it
costs in review load.

## Arms (same evidence, no model call)

| id | gate-unknown branch |
| --- | --- |
| **A (frozen)** | `compliant`, as in every reported result |
| **B (abstain)** | `need_review` |

Both arms recompute `symbolic_verdict` over the Stage-4 evidence stored in the frozen judge
cache `results/2026-09-22_rcasr_judge/{internal,public}_pairs.jsonl`. Arm B changes only
the defaults table, and only the gate-unknown branch reads that table. Pairs whose stored
evidence is `None` keep their stored label in both arms.

## Checks, fixed in advance

1. **Sanity check (exact, required).** Arm A must reproduce the stored label on every pair
   with evidence. Any mismatch aborts the analysis: the cache and the code would then
   disagree.
2. **Primary: headline neutrality (exact, required).** The set of `non_compliant` pairs is
   identical under A and B, on all internal and all public cached pairs. Every grounded
   violation metric reads only `non_compliant`: GV-R, closed-world GV-P, GV-F1 and false
   alarms per image (`rcasr_experiment.e2e_metrics`). So neutrality means every reported
   E2, low-width E2 and E5b number is unchanged.
3. **Secondary (descriptive, no gate).**
   - The `need_review` rate among judged pairs under A and B:
     - the RCASR-BM25 width-2 arm and the width-3 arm (internal);
     - all public pairs.
   - The number of re-routed pairs whose gold status is `non_compliant`, meaning
     violations that arm A labelled `compliant` and arm B sends to a reviewer. This is
     internal gold only, and it is descriptive.

## Decision rule

- **If checks 1 and 2 both hold:** the stabilised and deployed configuration adopts
  **B**, and `deploy/` implements it. The secondary numbers are reported as its cost.
  The manuscript may state that the §3.4 default is headline-neutral by construction and
  verified on the cache, and that the deployed system abstains on unassessed subjects.
- **If either check fails:** the deployed judge stays at A, and the failure is recorded.

There is no threshold on the secondary numbers: adoption does not depend on the review
cost, which is only reported. This is not a new method arm and moves no fitted quantity.
