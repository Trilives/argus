# RCASR on ARGus

Code, data and research artifacts for **Risk-Controlled Applicability Set Retrieval for
Regulation-Grounded Construction-Site Visual Screening**.

RCASR selects which provisions apply to a site photograph. It combines a retrieval prior with
posteriors over 131 visual atoms, selecting a set under asymmetric miss/exposure costs at a target
width or under conformal risk control. ARGus supplies screening, retrieval agents and symbolic
judgements that trace back to regulatory clauses.

The paper contributes the applicability selector, site-grouped evaluation design and failure
analysis. Company-owned photographs provide the validation platform; a new dataset is not a
claimed contribution. See [the manuscript-to-repository map](docs/PAPER_ALIGNMENT.md).

**Image use:** 463 additional files are **non-commercial only (CC BY-NC 4.0)**. The gold images
retain CC BY 4.0, and 37 previously published photographs in the additional subset retain their
existing grants. Check each image's `license` field in the [manifest](data/image_pool/manifest.csv)
and the [image licence](data/research_images/LICENSE) before reuse. Software remains Apache-2.0.

## Start here

| Goal | Entry point |
|---|---|
| Explore documentation | [Documentation index](docs/README.md) |
| Read research plans in order | [Research plans and experiment map](docs/research/README.md) |
| Find results and their files | [Results overview](docs/RESULTS_OVERVIEW.md) · [Result files](results/README.md) |
| Inspect the image pool | [Data card](data/README.md) · [Combined manifest](data/image_pool/README.md) |
| Run inference | [Examples](examples/README.md) · [Models](MODELS.md) |
| Verify scores and frozen inputs | [Reproducibility](docs/REPRODUCIBILITY.md) |
| Understand company involvement and ownership | [Research disclosures](docs/RESEARCH_DISCLOSURES.md) |
| Verify the repository | [Verification guide](docs/VERIFICATION.md) |

## Install and verify

Run from the repository root with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run --locked python scripts/verify_release.py
uv run --locked --with pytest python -m pytest -q tests
uv run --locked python scripts/bootstrap_rule_library.py --check data/rules/rules_en.json
```

The offline verifier checks image hashes/subsets/metadata, release versions, local documentation
links, known snapshot exceptions and the complete public-file manifest. Tests require no models
or private credentials; tests requiring restricted inputs are explicitly skipped.

Recompute the included ARGus baseline metric from bundled labels and predictions:

```bash
uv run --locked python reproduce/score_grounded_violations.py \
  reproduce/gold_labels.json reproduce/predictions/R4_image_rules.jsonl
```

This yields **GV-F1 0.514** (430 true positives, 982 predictions, 691 gold violations).
See [label-only reproduction](reproduce/REPRODUCE.md) for the freeze and reliability report.

## Run RCASR

Serve a model with token log-probabilities using the [model instructions](MODELS.md), then:

```bash
export RCASR_EXTRACTOR_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
uv run --locked python run_rcasr.py examples/sample_images/P001.jpg --no-judge
uv run --locked python run_rcasr.py data/research_images/img_0004.jpg
```

These commands make model calls to your configured server. The [example guide](examples/README.md)
covers thresholds, judgement, the ARGus demo and gated third-party inputs. Optional dependency
groups are `serving` and `analysis`; install them for the workflows that need them.

## Included data

| Subset | Files | Labels and purpose |
|---|---:|---|
| [Gold originals](data/gold_images/README.md) | 500 | Unchanged evaluation gold; label-only annotations in `reproduce/` |
| [Additional originals](data/research_images/README.md) | 500 | Privacy-reviewed research inputs; no new gold labels |
| [Demonstration files](examples/README.md) | 40 | Existing anonymised examples; disjoint from gold |

The combined original-image pool has **1,000 anonymous IDs**, representing 968 distinct source
byte streams and 32 alias pairs. It follows a fixed selection, rather than uniform random
sampling or a new annotated test design. **No cropped dataset images are included.**

The [manifest](data/image_pool/manifest.csv) separates input `source_sha256` from anonymised-file
`sha256`. All additions were reviewed individually: 183 have blur regions and 317 were saved with
no blur needed. All 1,000 original JPEGs are at most 2048 px on the longest side and contain no EXIF/GPS metadata.
Unblurred inputs, full evidence, site keys, robot frames and gated third-party imagery/labels
remain restricted. [Reproducibility](docs/REPRODUCIBILITY.md) explains the reproduction limits.

## Research scope

The primary comparison uses 346 photographs from 48 sites under 42 provisions, with site-grouped
cross-fitting. At matched width 3, recorded-violation recall rose from 0.515 (BM25) to 0.709.
The later width-2 arm kept BM25's width-3 recall while false alarms fell from 1.28 to 1.00 per image.
Its timing differs from the primary plan; revision analyses H1–H8 are post hoc. See
[analysis timing](docs/RESEARCH_DISCLOSURES.md#analysis-timing) and [results](results/README.md).

## Licence and citation

Code and rule representation use [Apache-2.0](LICENSE). Gold photographs and labels retain
CC BY 4.0. The [additional-image licence](data/research_images/LICENSE) restricts 463 eligible
files to non-commercial use and preserves the existing grants for listed exceptions. Company ownership, the first
author's authorization and delegated upload, funding and involvement are described in
[research disclosures](docs/RESEARCH_DISCLOSURES.md) and [licensing](docs/LICENSING.md).

Cite the manuscript and existing archive [10.5281/zenodo.21535053](https://doi.org/10.5281/zenodo.21535053).
[CITATION.cff](CITATION.cff) contains the author list and manuscript citation.
