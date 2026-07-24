# Examples

A self-contained set to test-run and understand the pipeline without the full dataset.

## `sample_images/` — 20 real construction-site images

Twenty privacy-reviewed site photographs (`P001.jpg` … `P020.jpg`), disjoint from the
evaluation gold set, provided so you can run the pipeline end to end on real inputs. They were
manually reviewed to contain no identifiable individuals or label-bearing markings; SHA-256
checksums and dimensions are in `sample_images/manifest.csv`. These images carry no gold labels
(they are a demonstration set, not an evaluation set).

## Test-run the pipeline

The turnkey path is the demo runner at the repo root, which downloads + serves a model and screens
these images automatically:

```bash
uv sync --group serving
uv run python run_demo.py            # see run_demo.py for env-var knobs (model, retriever, count)
```

If you already run an OpenAI-compatible endpoint, point the demo at it instead of serving one:

```bash
export OPENAI_BASE_URL="http://<your-host>/v1"
export OPENAI_API_KEY="<key-or-EMPTY-for-vLLM>"
uv run python run_demo.py
```

No endpoint needed to explore the rule library or author rules:

```bash
uv run python scripts/bootstrap_rule_library.py --check data/rules/rules_en.json
uv run python scripts/bootstrap_rule_library.py --template
```

## Expected output shape

`expected_output_P001.json` is a **hand-authored illustrative** screening record for `P001.jpg`
(an opening-protection scene) — it is not a captured model run, and its purpose is to show the
artifact schema: the fact record, the retrieved candidate rules, the per-checkpoint evidence, the
per-clause verdict, and the rectification advice. A real run produces the same shape with values
decided by the configured models.
