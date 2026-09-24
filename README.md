# Risk-Controlled Applicability Set Retrieval for Regulation-Grounded Construction-Site Visual Screening

**RCASR** (on the **ARGus** framework): code, frozen configurations, analysis plans and aggregate
results for the paper of that name.

Screening a site photograph against a regulation library first requires deciding which provisions
apply. Fixed top-*k* retrieval ranks provisions by textual relevance, not visible applicability, so
widening the list buys recall only by exposing the judge to inapplicable provisions. RCASR frames
applicability as **set prediction under asymmetric miss/exposure cost**. It re-weights a retrieval
prior by the probability that each provision's compiled subject and state conditions hold, computed
from token-level posteriors over 131 rule-agnostic visual atoms with abstention. It then selects a
set either at a target mean width (budget mode) or under conformal risk control on the miss rate.

Headline results (all pre-specified; the internal evaluation uses site-grouped cross-fitting):

- **346 photographs, 48 sites, 42 provisions, matched mean width 3.** Recorded-violation recall
  rose from 0.515 (BM25) to 0.709. RCASR beat an unstructured adaptive threshold and a ternary
  screen, and matched a multi-turn agent retriever.
- **Width-2 arm.** With one frozen judge, a pre-specified width-2 arm kept BM25's width-3 recall.
  False alarms per image fell from 1.28 to 1.00.
- **ConstructionSite-10k (public).** Frozen before public scoring, RCASR raised violation detection
  for harness use from 0.08 to 0.80 and for PPE from 0.43 to 0.67. It lost the excavator-radius
  rule; the mechanism is reported in the paper.
- **Conformal risk control.** It held its miss-risk target, at a price of about ten candidates per
  image at α = 0.10.
- **Patrol robot.** The configuration runs on a quadruped patrol robot. The deployed runtime is
  checked byte for byte against this analysis code, and one stored site round is reported with
  operating proxies (no field-accuracy claim).

> **Status.** Research code accompanying the manuscript above. Release **v2.0.0** adds RCASR on
> top of the ARGus screening framework of the earlier study (releases v1.x), whose code, results and
> online supplement are kept unchanged below.

## RCASR: what was added in v2.0.0

```
src/
├── rcasr.py                 # structured applicability score + width / CRC selection
├── atom_posterior.py        # token-level atom posteriors with abstention (two rule-agnostic passes)
├── typed_schema.py, typed_predicates.py, typed_evidence.py   # provision formulas compiled over atoms
├── rcasr_experiment.py      # site-grouped cross-fitting, arms, ablations, gates, CRC calibration
├── site_partition.py        # site units and folds (near-duplicate clusters fused)
├── set_selection*.py        # set-selection metrics and matched-width reporting
├── public_map.py            # frozen mapping of ConstructionSite-10k rules to the library
└── research_snapshot.py     # SHA-256 input snapshots and their verification
experiments/retrieval/
├── eval_rcasr.py            # internal evaluation (evidence → cross-fitted selection)
├── judge_rcasr.py, judge_rcasr_shard.py   # frozen end-to-end judge over the selected sets
├── eval_lowwidth_e2.py      # the pre-specified width-2 arm
├── eval_rcasr_public.py     # public auxiliary evaluation (ConstructionSite-10k)
└── eval_s34_default.py      # the no-subject abstention default adopted by the deployed judge
configs/
├── rcasr_runtime_v1.json              # the frozen configuration the robot runs (sha256 ea624e49…)
└── rcasr_runtime_v1_recal_site.json   # its label-free, site-recalibrated threshold (sha256 4cad9da1…)
data/rules/proposed/rcasr_v1.json      # frozen RCASR hyper-parameters and pre-specified grid
data/rules/proposed/typed_evidence_v1.json   # the 131-atom vocabulary and per-provision bindings
data/rule_assets/                      # rule units and the BM25 index built from the library
data/public_eval/public_map_v1.json    # public rule mapping (the public labels are not redistributed)
docs/paper_v3/design/*.md              # analysis plans, each written before its run
results/2026-09-2*/                    # aggregate results + input snapshots for every v3 table
```

**Verify the freeze.** Each `results/2026-09-2*/input_snapshot.json` lists the SHA-256 of every input
of that run. Paths are kept as in the research repository, so they resolve here. The analysis plans,
frozen configurations, rule assets, experiment scripts and RCASR modules match their recorded hashes
byte for byte (75 entries). The rest do not verify here, each for a stated reason:

- **Restricted inputs are not released:** gold labels, images, site keys, scene facts, atom
  evidence and per-pair verdicts.
- **Files belong to this public package, not to the research checkout:** `pyproject.toml`,
  `uv.lock`, and `src/config.py` / `src/schemas.py` (comments only).
- **Redacted:** `results/2026-09-22_rcasr_v2/rcasr_results.json` (see Tables below).
- **Re-serialised JSON:** `results/data_audit/near_duplicates.json` is the v1.x copy, with the same
  content in different formatting.
- **Skip guard added:** `tests/test_site_partition.py` skips when the restricted site keys are
  absent.

`configs/rcasr_runtime_v1.json` names the snapshots it was derived from, and those hashes verify
against the files here.

**Tables.** `results/README.md` maps every table of the paper to the file behind it. The internal
result `results/2026-09-22_rcasr_v2/rcasr_results.json` is released with construction-project names
replaced by `site_NN` and lost gold pairs reduced to counts, so its hash differs from the research
copy.

**Tests.** `python -m unittest discover -s tests` runs 118 tests without models or restricted data;
8 skip because they need the restricted site keys or the ConstructionSite-10k labels.

## The ARGus framework (earlier study, releases v1.x)

The screening framework RCASR plugs into. Given a site photograph, a bounded retrieval **agent**
proposes the few provisions that plausibly apply. The verdict is then computed from a named provision
formula over persisted checkpoint evidence, never the agent's free text, so every decision traces
back to a clause. The earlier study measured where such screening fails: handed the correct
provision, the judge reaches a grounded-violation F1 of **0.867**, while the best complete
configuration reaches **0.514**. Retrieval, deciding which provisions apply, dominated end-to-end
accuracy, and grounded precision fell at every widening step measured
([`results/e2e/k_scan_summary.md`](results/e2e/k_scan_summary.md)). That finding motivates RCASR.
The earlier study's online supplement is [`paper/supplement.pdf`](paper/supplement.pdf).

## What's here

```
argus/
├── src/                         # the framework (rule-set-agnostic)
│   ├── retrieval/               # fixed retrievers + the grep/read/submit agent (R1–R4)
│   │   ├── agent_grep.py        #   text agent (R3) + the bounded loop / anytime termination
│   │   ├── agent_grep_visual.py #   image-conditioned agent (R4)
│   │   ├── bm25.py, text_overlap.py, siglip.py
│   ├── pipeline.py              # the five-stage screening pipeline
│   ├── symbolic_judgement.py    # neuro-symbolic (Kleene) verdict (J3-sym)
│   ├── rules.py, schemas.py, validation.py, rule_families.py
│   ├── backends/                # OpenAI-compatible (vLLM/SGLang) + local backends
│   └── config.py, paths.py, ...
├── scripts/
│   └── bootstrap_rule_library.py  # interactive, schema-validating rule authoring tool
├── data/
│   ├── rules/rules_en.json      # the instantiated rule library (42 rules)
│   ├── rules/rules_schema_en.json
│   └── README.md                # gold data card (evaluation set is on request)
├── examples/                    # 20 privacy-reviewed sample images + a runnable walkthrough
├── paper/supplement.pdf         # the paper's online supplement (ASCE no longer hosts these)
├── results/                     # aggregate metrics behind the paper's tables
├── reproduce/                   # freeze proof + label-only gold + GV scorer (verify the numbers)
├── tests/                       # hermetic unit tests for the core invariants
├── run_demo.py                  # end-to-end demo (API or local vLLM service)
├── MODELS.md                    # evaluated models + serve commands
├── Prompts_en/                  # stage prompts (English runtime)
├── pyproject.toml               # uv project (core deps + optional `serving` group)
├── LICENSE / NOTICE / CITATION.cff
```

Run the tests (no models needed): `python -m unittest discover -s tests`.

## Install

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # core deps (read the code, validate/author rules)
uv sync --group serving  # + model-serving deps, only to run the VLM/judge pipeline
```

The core sync is enough to read the code, validate the rule library, and use the authoring tool.
Running the VLM/judge pipeline additionally needs an OpenAI-compatible endpoint and the `serving`
dependency group. Prefix commands with `uv run` (e.g. `uv run python scripts/...`).

## Quick start

**Run the end-to-end demo.** One multimodal model (default **Qwen3.5-9B**) runs every stage — image
facts, the retrieval agent, and the judgement — over the bundled sample images. Two modes:

```bash
# Local service (needs a GPU): serves local Model/ weights if present, else downloads Qwen/Qwen3.5-9B
uv sync --group serving
uv run python run_demo.py

# API mode: use an existing OpenAI-compatible endpoint (no local serving)
OPENAI_BASE_URL=http://host:8000/v1 OPENAI_API_KEY=EMPTY uv run python run_demo.py
```

Serve flags come from the model card (`--tool-call-parser qwen3_coder`). Configure via env vars
(`ARGUS_MODEL`, `ARGUS_MODEL_PATH`, `ARGUS_RETRIEVAL`, `ARGUS_NUM_IMAGES`, …; see `run_demo.py`). Results
land in `results/demo/`.

**Author or validate rules (no models needed).**

```bash
# Validate the shipped library
uv run python scripts/bootstrap_rule_library.py --check data/rules/rules_en.json

# Print a blank rule template
uv run python scripts/bootstrap_rule_library.py --template

# Interactively add a rule to a new library for another regime
uv run python scripts/bootstrap_rule_library.py --path data/rules/my_regime.json --new
```

The tool enforces the schema plus the cross-field invariants (e.g. `required_checkpoints`
mirrors the `visual_checkpoints` keys; `rule_id` uniqueness), and writes atomically.

**Run screening (needs an endpoint).** Point the pipeline at an OpenAI-compatible server via
environment variables — never commit these:

```bash
export OPENAI_BASE_URL="http://<your-host>/v1"
export OPENAI_API_KEY="<key-or-EMPTY-for-vLLM>"
```

Model and stage configuration lives in `src/config.py`.

## The rule unit

A regime is onboarded by decomposing it into rule units of one schema — not by retraining.
Each unit carries provision provenance (`source_level`, `source_quote`), atomic
`visual_checkpoints`, a boolean `visual_screening_rule` over those checkpoints, a
`decision_scope` (so quantities beyond a single image route to review rather than being
guessed), retrieval text, and `rectification_advice`. See `data/rules/rules_schema_en.json`.

## Data availability

- **Included:** all source code (ARGus and RCASR), the rule library and schema, the RCASR atom
  vocabulary and frozen configurations, the analysis plans, the stage prompts, 20 privacy-reviewed
  **sample images** (`examples/sample_images/`, no gold labels), and the **aggregate result
  summaries** and input snapshots behind the papers' tables (`results/`).
- **Not redistributed:** the ConstructionSite-10k labels and images (CC BY-NC 4.0, gated access from
  their authors). Only the frozen rule mapping and RCASR's per-image selections on the public test
  images are included.
- **On reasonable request:** the full construction-site image pool and the gold image–rule
  annotations. Site imagery contains identifiable people and is governed by a site-operator
  agreement, so it is released under a data-use agreement, not publicly. See `data/README.md`.

## Reproducibility

`reproduce/` lets a reviewer verify the central claims from released, non-sensitive files:

- **Pre-registration:** `freeze_manifest.json` is a timestamped SHA-256 freeze of the gold + rules —
  proof the labels were fixed before the runs.
- **Headline metric, exactly:** `python reproduce/score_grounded_violations.py reproduce/gold_labels.json
  reproduce/predictions/R4_image_rules.jsonl` reproduces the best cell **GV-F1 0.514**
  (tp 430 / pred 982 / gold 691) from label-only gold + predictions — no images needed.
- **Reliability:** `reproduce/reliability_agreement.json` is the label-only inter-annotator report
  (exact-match 0.77, Jaccard 0.906, κ 0.948) with the full disagreement list.

See `reproduce/REPRODUCE.md`. Images, scene facts, and evidence text stay on request (privacy).

## Licensing, provenance, and redistribution

Source code and the rule representation are released under the **Apache License 2.0** (`LICENSE`,
`NOTICE`).

We audited the bundle for redistribution before release:

- **No standard documents are included.** The repository ships no PDF or verbatim copy of any GB /
  JGJ / Shenzhen standard.
- **`source_quote` fields are paraphrased English summaries** of each requirement (median ~200
  chars), not verbatim clause transcriptions — they describe the rule *logic*, which is fact-like
  and distributable, rather than reproducing a copyrighted clause. The field name is historical;
  treat its contents as summaries, not official text. Apache-2.0 covers this representation; it
  asserts no rights over the underlying standards, which remain the property of their issuing
  bodies.
- **Prompts** (`Prompts_en/`) contain no embedded standard text.
- **Author-confirmation items** (see checklist): a few rules are derived from an internal source
  (`source_level` = "CSCEC Third Bureau Shenzhen public source"): `R-BHV-007`, `R-CIV-016`. Confirm
  these are cleared for public release. Engineering-derived and accident-report-derived rules are
  the authors' original work.

If any specific summary is judged too close to a protected clause, replace it with a looser
paraphrase of the rule logic before publishing.

## Citation

A BibTeX entry will be added on acceptance. For now, cite the manuscript "Risk-Controlled
Applicability Set Retrieval for Regulation-Grounded Construction-Site Visual Screening" and the
archive [10.5281/zenodo.21535053](https://doi.org/10.5281/zenodo.21535053).

That is the **all-versions** DOI: it always resolves to the latest release, which is what the
paper's Data Availability Statement cites. To cite one exact snapshot, use the version DOI shown on
that release's Zenodo page instead.

---
### Release checklist (run before tagging a new version)
- [ ] Re-run `mirror_public_results.py` in the working repo so `results/` matches the paper (it also
      copies the v3 configs, analysis plans and rule assets byte for byte).
- [ ] Skim `source_quote` fields; loosen any summary judged too close to a protected clause.
- [ ] Confirm no images / annotations / per-pair dumps / `.env` are staged — `.gitignore` covers
      them, but verify with `git status` rather than trusting it.
- [ ] Refresh `paper/supplement.pdf` if the manuscript's supplement changed.
- [ ] Bump `version` in `CITATION.cff` and `.zenodo.json`, then tag; Zenodo mints the version DOI.
