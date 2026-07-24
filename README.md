# ARGus — Agentic Rule-Grounded Screening

Reference implementation for an auditable framework that screens an image against a large,
visually confusable regulatory rule set while keeping every decision traceable to a clause.
A bounded retrieval **agent** proposes the few candidate rules that plausibly apply; a
**human-auditable** verdict is then a function of a named rule formula and persisted checkpoint
evidence — never the agent's free text. The agent proposes; the rule formula and a human
inspector dispose.

> **Status.** Research code accompanying a manuscript under review at *Advanced Engineering
> Informatics* (Special Issue on Trustworthy Hybrid Human Agentic-AI Engineering Systems).
> This repository is provided so reviewers can inspect the method and the rule schema.

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

- **Included:** all source code, the rule library and schema, the stage prompts, 20 privacy-reviewed
  **sample images** (`examples/sample_images/`, no gold labels), and the **aggregate result
  summaries** behind the paper's tables (`results/`).
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

A BibTeX entry will be added on acceptance. For now, cite the manuscript
"Agentic Rule Retrieval for Traceable, Rule-Grounded Visual Compliance Screening against Complex
Regulatory Rule Sets" (under review).

---
### Before you make this public — checklist
- [ ] Fill in the copyright holder in `NOTICE` and the author/citation block above.
- [ ] Skim `source_quote` fields; loosen any summary judged too close to a protected clause.
- [ ] Re-confirm no images / annotations / results / `.env` are staged (`.gitignore` covers them, but verify `git status`).
- [ ] Optional: mint the Zenodo DOI and add it here and to the paper's Data Availability statement.
