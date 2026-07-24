"""Path constants for the CS evidence-chain line.

The CS line is **self-contained**: every runtime path resolves under ``CS/``.
Shared root assets (rules, splits, annotations, images) are copied into
``CS/data/`` by ``CS/experiments/sync_shared_assets.py``; nothing here reaches
back into the repository root at runtime. ``shared/rules/rules.json`` remains the
canonical source — edit it there first, then re-sync (see that script).

Language: the runtime pipeline is **English**. ``RULES_PATH`` points at the
English rule library (``rules_en.json``) and ``PROMPT_DIR`` at ``Prompts_en/``.
The Chinese library (``rules.json``) and ``Prompt/`` are kept as the maintained
bilingual counterparts (field- and id-aligned) but are not loaded at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

CS_ROOT = Path(__file__).resolve().parents[1]  # CS/src/paths.py -> CS/
PROJECT_ROOT = CS_ROOT.parent  # only used by the offline sync script, never at runtime

# --- top-level CS directories -------------------------------------------------
DATA_DIR = CS_ROOT / "data"
RESULTS_DIR = CS_ROOT / "results"
TMP_DIR = CS_ROOT / "tmp"
MODEL_ROOT = CS_ROOT / "Model"
PROMPT_DIR = CS_ROOT / "Prompts_en"

# --- CS-local copies of shared assets (see sync_shared_assets.py) -------------
IMAGES_DIR = DATA_DIR / "images"
# English is the runtime rule library; rules.json (Chinese) is the bilingual
# counterpart, id-aligned but not loaded at runtime.
# Ablation overrides. Default to the canonical assets; an experiment that needs a
# variant sets the env var for its own process only, so arms can run concurrently
# against one checkout instead of editing shared files while another run is live.
RULES_PATH = Path(os.environ.get("CS_RULES_PATH") or (DATA_DIR / "rules" / "rules_en.json"))
RULES_ZH_PATH = DATA_DIR / "rules" / "rules.json"
SPLITS_DIR = DATA_DIR / "splits"
IMAGE_MAPPING_PATH = DATA_DIR / "annotations" / "image_mapping.csv"

# --- derived rule assets (built by experiments/build_rule_assets.py) ----------
RULE_ASSETS_DIR = DATA_DIR / "rule_assets"
RULE_UNITS_PATH = RULE_ASSETS_DIR / "rule_units.json"
RULE_INDEX_PATH = RULE_ASSETS_DIR / "rule_index.json"
RULE_STATS_PATH = RULE_ASSETS_DIR / "rule_stats.json"
RULE_FIELDS_PATH = RULE_ASSETS_DIR / "rule_fields.md"

# --- curated evidence-chain demo assets ---------------------------------------
CHAIN_DIR = DATA_DIR / "chain"
EVIDENCE_CHAIN_SCHEMA_PATH = CHAIN_DIR / "evidence_chain_schema.json"
EVIDENCE_CHAIN_SAMPLES_PATH = CHAIN_DIR / "evidence_chain_samples.json"
VIOLATION_INSTANCES_SCHEMA_PATH = CHAIN_DIR / "violation_instances_schema.json"

# --- prompts --------------------------------------------------------------
# One subdirectory per method, plus shared/ for prompts reused across methods
# (see docs/MODULARITY.md's directory contract for Prompt/).
PROMPT_SHARED_DIR = PROMPT_DIR / "shared"
PROMPT_CHAIN_DIR = PROMPT_DIR / "chain"
PROMPT_DECOUPLED_DIR = PROMPT_DIR / "decoupled"
PROMPT_RULE_REFINE_DIR = PROMPT_DIR / "rule_refine"

# shared/: Stage 1 fact extraction, reused by every JUDGEMENT_MODE
IMAGE_FACT_PROMPT_PATH = Path(
    os.environ.get("CS_FACT_PROMPT") or (PROMPT_SHARED_DIR / "image_fact_prompt.md")
)
SCENE_FACT_PROMPT_PATH = PROMPT_SHARED_DIR / "scene_fact_prompt.md"

# chain/: the one-pass judgement family — legacy hybrid (JUDGEMENT_MODE="chain")
# plus its two clean protocol splits J1/J2 (same output schema, different inputs)
EVIDENCE_CHAIN_PROMPT_PATH = PROMPT_CHAIN_DIR / "evidence_chain_prompt.md"
FACTS_RULES_PROMPT_PATH = PROMPT_CHAIN_DIR / "facts_rules_prompt.md"    # J1: facts+rules, no image
IMAGE_RULES_PROMPT_PATH = PROMPT_CHAIN_DIR / "image_rules_prompt.md"    # J2: image+rules, no facts

# decoupled/: per-rule evidence extraction + text-only judgement (Stage 4/5,
# JUDGEMENT_MODE="decoupled")
RULE_EVIDENCE_PROMPT_PATH = PROMPT_DECOUPLED_DIR / "rule_evidence_prompt.md"
# strict variant: subject-gate checkpoints judged against the rule's exact
# subject type (the J3-sym-strict arm); forked so cached pairs stay prompt-bound
RULE_EVIDENCE_STRICT_PROMPT_PATH = PROMPT_DECOUPLED_DIR / "rule_evidence_strict_prompt.md"
RULE_EVIDENCE_BATCH_PROMPT_PATH = PROMPT_DECOUPLED_DIR / "rule_evidence_batch_prompt.md"
JUDGEMENT_PROMPT_PATH = PROMPT_DECOUPLED_DIR / "judgement_prompt.md"

# rule_refine/: aggregate-then-filter retrieval refinement (Stage 2.5a/b/c,
# RULE_REFINE_MODE="aggregate_filter")
RULE_AGGREGATE_PROMPT_PATH = PROMPT_RULE_REFINE_DIR / "rule_aggregate_prompt.md"
RULE_AGGREGATE_OBSERVATION_PROMPT_PATH = PROMPT_RULE_REFINE_DIR / "rule_aggregate_observation_prompt.md"
RULE_FILTER_PROMPT_PATH = PROMPT_RULE_REFINE_DIR / "rule_filter_prompt.md"

# --- result directories -------------------------------------------------------
RETRIEVAL_RESULTS_DIR = RESULTS_DIR / "retrieval"
PIPELINE_RESULTS_DIR = RESULTS_DIR / "pipeline"

# Source-of-truth paths for the offline sync script only (repo root).
SHARED_RULES_PATH = PROJECT_ROOT / "shared" / "rules" / "rules.json"
SHARED_RULES_EN_PATH = PROJECT_ROOT / "shared" / "rules" / "rules_en.json"
SHARED_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
SHARED_IMAGE_MAPPING_PATH = PROJECT_ROOT / "data" / "annotations" / "image_mapping.csv"
SHARED_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
