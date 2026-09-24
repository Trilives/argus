# Examples

A self-contained set to test-run and understand the pipeline without the full dataset.

## `sample_images/`: 40 real construction-site images

Forty site photographs (`P001.jpg` … `P040.jpg`), disjoint from the evaluation gold set, provided
so you can run the pipelines end to end on real inputs. They carry no gold labels: this is a
demonstration set, not an evaluation set. SHA-256 checksums and dimensions are in
`sample_images/manifest.csv`.

- **P001–P020** were chosen to contain no identifiable individuals or label-bearing markings,
  and were reviewed by hand.
- **P021–P040** include workers, so they are anonymised with a Gaussian blur. Each image blurs:
  - every head found by a hard-hat/head detector, at low confidence, with the box padded by 35%;
  - the top 30% of every person found by a COCO person detector, as a second net for missed heads;
  - hand-drawn boxes over text that could identify a company, site or device: names on vests,
    brand names on formwork, posted documents, site signage and a camera watermark. Each box was
    drawn while viewing the image at full resolution.

  Every region gets a two-pass Gaussian blur with sigma = max(8 px, 0.25 × the region's shorter
  side), so larger regions get a stronger blur. The export checks the pixels and refuses to write
  an image that fails:
  - in every region, the residual detail (variance of the Laplacian) must fall to at most 0.10 of
    the original. The largest ratio across the 20 images is 0.025;
  - every pixel outside the regions must be unchanged. The count is 0 for all 20;
  - the file is re-encoded from pixels (JPEG q90) and read back to confirm it carries no
    EXIF, ICC profile or comment.

  `sample_images/anonymisation.json` records the region count, their kinds and these checks for
  each image. Generic safety slogans (for example 当心坠落, "beware of falling") are left
  readable because they are part of what a screener sees.

## RCASR guide: screen images with the frozen configuration the robot runs

`run_rcasr.py` runs RCASR per image with the same functions that produced the paper's numbers:
1. Stage-1 scene facts and two token-log-probability atom passes (subject and state).
2. A BM25 prior over the facts.
3. The frozen structured score and a global threshold. An empty set is a legal outcome.
4. Optionally, the symbolic judge on each selected provision.

**1. Serve the extractor.** RCASR reads token log-probabilities, so it needs an OpenAI-compatible
server that returns them. vLLM does. Serve Qwen3.5-9B as in `MODELS.md`:

```bash
uv sync --group serving
vllm serve Qwen/Qwen3.5-9B --port 8000 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --max-model-len 32768
export RCASR_EXTRACTOR_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY
```

The judge uses the same server unless `RCASR_JUDGE_URL` / `RCASR_JUDGE_MODEL` point elsewhere.
Endpoints are read from the environment and never written to disk.

**2. Screen the sample images.**

```bash
uv run python run_rcasr.py examples/sample_images                # threshold of the robot (width 2)
uv run python run_rcasr.py examples/sample_images --no-judge     # selection only, no judge calls
```

Each image prints one line: its decision (`violation`, `need_review`, `clear` or `clear_empty`),
then each selected provision with its verdict. `need_review` means the judge abstained. One cause
is a pair routed to the no-subject default, which on the robot abstains instead of reporting
`compliant`. The full record goes to `results/demo_rcasr/rcasr_runs.jsonl` (git-ignored). It holds
the facts, the eight highest provision scores, the selected set and the verdicts.

**3. Screen public images and compare with the released selection.** The ConstructionSite-10k
test images are gated on Hugging Face (CC BY-NC 4.0, non-commercial research only), so this
repository does not include them. With an account that has accepted the dataset's terms:

```bash
uv sync --group analysis
export HF_TOKEN=<your read token>
uv run python scripts/fetch_public_images.py              # the 8 guide images, SHA-256-checked
uv run python run_rcasr.py examples/public_images --threshold public --check-public
```

`fetch_public_images.py` writes to `examples/public_images/` (git-ignored). It checks that each
image is byte-identical to the input recorded in
`results/2026-09-22_rcasr_public/input_snapshot.json`. `--ids` fetches any other test image of the
public evaluation. `--threshold public` is the width-3 threshold of the public run. From the run's
recorded facts and atom evidence, it reproduces the released
`results/2026-09-22_rcasr_public/public_selection.json` on all 412 images.

| Image | Public rule | What the frozen run recorded |
|---|---|---|
| 0000068, 0000095 | rule_1 (PPE) | RCASR's set let the judge confirm the violation; matched-width BM25 missed it |
| 0000328, 0001392 | rule_2 (harness at height) | as above |
| 0000009, 0000626 | rule_3 (edge protection) | as above |
| 0000327, 0000331 | rule_4 (excavator radius) | the failure the paper reports: RCASR drops the provision |

`--check-public` prints each selection's Jaccard overlap with the released one. Expect close
agreement, not identity. The atom posteriors reproduce closely: on the robot's runtime they
differ from the recorded ones by about 0.002. The Stage-1 facts, however, are free text. A small
numeric difference from another GPU or batch can change their wording, which moves the BM25 prior
and can change a provision near the threshold. In our checks on these 8 images, 4 and then 5 of
the 8 sets matched exactly, and the mean Jaccard was about 0.8. Treat the released selection as
the reference, and a fresh run as a demonstration of the same method.

## ARGus demo (earlier study)

The turnkey demo runner at the repo root serves a model and screens these images with the ARGus
retrieval agent:

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

`expected_output_P001.json` is a **hand-authored illustrative** screening record for `P001.jpg`,
an opening-protection scene. It is not a captured model run. It shows the ARGus artifact schema:
- the fact record;
- the retrieved candidate rules;
- the per-checkpoint evidence;
- the per-clause verdict;
- the rectification advice.

A real run produces the same shape, with values decided by the configured models.
