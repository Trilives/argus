"""Editable run controls for the CS evidence-chain experiments.

All experiment knobs live here rather than as CLI flags; the scripts under
``CS/experiments`` are thin wrappers around this configuration. The two axes the
CS line varies are the **retrieval method** (text-overlap RAG vs agentic grep)
and the **model backend** (local vLLM on GPU vs a remote OpenAI-compatible
endpoint), both selectable below.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

import paths

load_dotenv()

# === pipeline composition ====================================================
# Rule retrieval method compared head-to-head in the CS paper. The protocol
# (docs/EXPERIMENT_PROTOCOL.md §3, RQ1) names four retrieval systems R1–R4;
# R1/R3/R4 are selected through this knob, R2 (image-SigLIP2) is a separate axis
# built via build_siglip_retriever (see SIGLIP_* below). Canonical map is the
# authoritative PROTOCOL_RETRIEVAL_CODES dict further down this file.
#   R1 = text_overlap        (text-RAG over cached facts)
#   R3 = agent_grep          (text-agent-VLM over the same cached facts)
#   R4 = agent_grep_visual   (image-agent-VLM, raw image, no Stage 1 facts)
RETRIEVAL_METHOD = "agent_grep"  # "text_overlap" | "agent_grep" | "agent_grep_visual"

# Backend running the VLM stages (image-fact extraction + evidence-chain gen).
VLM_BACKEND = "openai_api"  # "local_vllm" | "openai_api"

# Master guard: no model is loaded or called unless this is True. Leave False
# for dry-run data preparation; flip on only when GPU / endpoint is free.
RUN_MODEL = False

# === local vLLM backend (GPU) ================================================
# Must be a vision model: VLM_BACKEND drives the image-fact + evidence-chain
# stages, so a text-only model (e.g. Qwen3.5-4B) cannot serve here.
LOCAL_MODEL_NAME = "Qwen3-VL-8B-Instruct"
LOCAL_MODEL_DIR = paths.MODEL_ROOT / LOCAL_MODEL_NAME

# === OpenAI-compatible endpoint backend ======================================
# vLLM/SGLang style server. Set OPENAI_BASE_URL in a local .env (gitignored) to
# point at your own endpoint; the placeholder below is not a working default.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")  # vLLM ignores the key, but the OpenAI client requires one.
OPENAI_TEXT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
# The Qwen3.6 endpoint now accepts images, so the same model serves the VLM
# stages — the whole pipeline can run on Qwen3.6. Small-model comparisons later.
OPENAI_VLM_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

# === agentic-grep retrieval ===================================================
# The retrieval agent is a text task, so it defaults to the remote endpoint
# regardless of VLM_BACKEND. Point AGENT_MODEL at any served text model to
# benchmark different models on the same grep loop.
AGENT_BACKEND = "openai_api"  # "openai_api" | "local_vllm"
AGENT_MODEL = OPENAI_TEXT_MODEL
# Env-overridable so the AGENT_MAX_ITERS sweep (2/4/8/16 -> cost-accuracy curve)
# can drive the loop cap without editing code. Default 8 reproduces reported runs.
AGENT_MAX_ITERS = int(os.environ.get("CS_AGENT_MAX_ITERS", "8"))
AGENT_MAX_GREP_HITS = 12
# Early-stop / graceful-degradation controls for the grep loop (see
# retrieval.agent_grep.run_agent_loop). When this many consecutive grep calls add
# no new rule_id (or merely repeat an already-run pattern), the loop stops probing
# and forces a best-effort submit instead of burning the rest of the iteration
# budget. On the final iteration the loop also forces a submit, so a run degrades
# to "anytime best-effort ranking" rather than "max_iters -> empty result".
AGENT_STALL_LIMIT = 3
# Qwen3.6 reasons before emitting tool calls; a submit turn can spend many tokens.
# Too small a budget truncates the turn (finish_reason=length) so no tool call is
# returned and the loop stalls to max_iters with an empty result (drove most of the
# ~13.5% agent failures in Records/2026-07-09). This is the PER-TURN OUTPUT cap, not
# the context window — the endpoint's max_model_len is ~96K (input+output+tools+image
# combined across the <=8 iterations), so keep this well below it. 8192 gives the
# reasoning+submit turn ample room while fitting small-model 32K serving contexts.
AGENT_MAX_TOKENS = 8192
AGENT_TEMPERATURE = 0.0

# === SigLIP-2 retrieval (image->rule similarity baseline vs agent) ===========
# Two input-resolution variants; the retriever code path is identical. Selected
# by key; embeds each rule's SIGLIP_TEXT_FIELD and ranks rules by image cosine.
SIGLIP_MODELS = {
    "224": "google/siglip2-base-patch16-224",
    "512": "google/siglip2-base-patch16-512",
}
SIGLIP_VARIANT = "224"  # "224" | "512"
SIGLIP_TEXT_FIELD = "visual_retrieval_text"

# === constrained decoding (vLLM structured outputs) ==========================
# When True, the plain-generation JSON stages pass their expected output schema
# (src/schemas.py) and the endpoint constrains decoding via the OpenAI
# ``response_format`` json_schema mechanism. Verified live against vLLM 0.25.1
# on the endpoint: ``response_format`` constrains, while the legacy
# ``guided_json`` extra_body is silently ignored there — do not switch back.
# Off by default because it changes the generation distribution: flip it per
# run and it is recorded in pipeline_manifest(). Tool-calling agent turns are
# unaffected (vLLM already schema-binds tool arguments).
GUIDED_JSON = False

# === shared generation knobs (recorded in manifests) ========================
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 1536
ENABLE_THINKING = False
# Greedy decoding (temperature 0) can fall into verbatim repetition loops — the
# VLM keeps padding a JSON array until it burns the whole token budget and the
# object never closes, so fact extraction yields no parseable JSON. A mild
# penalty breaks the loop; 1.3+ over-penalizes and truncates real content.
REPETITION_PENALTY = 1.1

# === pipeline driver =========================================================
# "samples": run over curated evidence_chain_samples (gold facts + gold rule;
#            retrieval scored head-to-head, chain stage optional).
# "split":   full image->facts->retrieve->chain over a dataset split (needs VLM).
PIPELINE_SOURCE = "split"

# === fact extraction mode (Stage 1) ==========================================
# "generic": one broad image-fact pass that feeds both retrieval and judgement
#            (baseline; the observed failure mode where generic facts get mined
#            into false-positive violations on compliant images).
# "scene":   retrieval-oriented lightweight scene facts — visible objects/parts/
#            materials/spatial relations/obvious state only, no verdict words.
FACT_MODE = "generic"  # "generic" | "scene"

# === judgement mode (Stage 4+5) ==============================================
# "chain":     current one-pass chain generation over candidate rules (baseline).
#              NOTE: this legacy mode feeds the model BOTH the raw image AND the
#              Stage 1 facts. The protocol (docs/EXPERIMENT_PROTOCOL.md §3, RQ2)
#              splits that hybrid into two cleaner isolated conditions — J1
#              (facts+rules, image=None) and J2 (image+rules, no facts) — which
#              are NOT yet implemented here; introducing them is Phase-B work
#              (needs model runs) and must not be silently faked. `chain` stays
#              the baseline until then.
# "decoupled": protocol J3 — per-rule rule-conditioned evidence extraction
#              (Stage 4, sees the image) then a text-only judgement over that
#              evidence (Stage 5), so the verdict is grounded in structured
#              checkpoint evidence rather than free reasoning over the raw image.
JUDGEMENT_MODE = "chain"  # "chain" (legacy hybrid) | "decoupled" (=J3)
# decoupled only: how many top candidate rules get their own evidence pass.
# top-1 vs top-k is the "single vs multi-candidate evidence" ablation.
EVIDENCE_TOP_K = 3
# decoupled only: when True, Stage 4 reads the image once for ALL candidate rules
# in a single call (per-rule checkpoint_evidence keyed by rule_id) instead of one
# call per rule. Fewer image encodings, and the model sees all candidates together.
EVIDENCE_BATCHED = True

# === protocol vocabulary map (docs/EXPERIMENT_PROTOCOL.md §3) ================
# Single source of truth linking the stable protocol codes (R1–R4 / J1–J3) to
# the implementation-level knob values above. The impl strings stay authoritative
# for code dispatch and on-disk result/cache filenames (renaming them would break
# every existing results/* file and Records/* provenance); these codes are the
# canonical labels used in the paper, docs tables and any result-tagging.
PROTOCOL_RETRIEVAL_CODES = {
    "R1": "text_overlap",        # text-RAG over cached facts
    "R2": "siglip",              # image-SigLIP2 (separate axis: build_siglip_retriever)
    "R3": "agent_grep",          # text-agent-VLM over the same cached facts
    "R4": "agent_grep_visual",   # image-agent-VLM, raw image, no Stage 1 facts
}
# J1/J2 are the clean splits of the legacy `chain` hybrid (implemented
# 2026-07-14: pipeline.judge_facts_rules / judge_image_rules share the chain
# output schema, differing only in inputs). `chain` itself stays available as
# the historical baseline but is NOT a protocol condition.
PROTOCOL_JUDGEMENT_CODES = {
    "J1": "facts_rules",  # facts+rules, image=None            — implemented
    "J2": "image_rules",  # image+rules, no Stage-1 facts      — implemented
    "J3": "decoupled",    # per-rule evidence -> text verdict  — implemented
}

# === rule refinement (Stage 2.5: aggregate-then-filter retrieval) ===========
# "none":            baseline; retriever.retrieve(top_k) goes straight to judgement.
# "aggregate_filter": retrieve a much larger BROAD_TOP_K_RULES set, have the model
#                     synthesize one aggregated observation prompt over their
#                     checkpoints (Stage 2.5a), re-read the image against that
#                     prompt for per-rule evidence (Stage 2.5b), then discard
#                     candidates unsupported by that evidence (Stage 2.5c). The
#                     one retrieval path where the image, not just the
#                     fact-mediated query, decides which rules survive.
RULE_REFINE_MODE = "none"  # "none" | "aggregate_filter"
# aggregate_filter only: how many candidates the initial broad retrieval keeps
# before Stage 2.5 narrows them back down.
BROAD_TOP_K_RULES = 15

# === asset language / cache isolation ========================================
# The runtime rule library and prompts are English (paths.RULES_PATH ->
# rules_en.json, paths.PROMPT_DIR -> Prompts_en/). Stage-1 facts, retrieval
# candidates and pair judgements produced by the earlier Chinese assets are NOT
# interchangeable with English runs: the facts are Chinese prose and the retrieval
# / pair caches carry the old Chinese rule_ids, which no longer join the migrated
# English gold. English runs therefore MUST use fresh caches. ASSET_LANG tags the
# shared Stage-1 facts cache so English and Chinese runs never share one file; the
# derived retrieval / oracle_pairs / e2e prediction working files must likewise be
# regenerated (or written under a language-tagged path) for an English run — do
# not resume them from the committed Chinese-era artifacts.
ASSET_LANG = "en"

# === dataset selection =======================================================
SPLIT_NAME = "development_subset"
LIMIT_IMAGES = 3
# Final candidate breadth. Capped at 3: a single site photo rarely triggers more
# than a few distinct rules at once, and a wider top-k mostly inflates RAG recall
# with off-scene rules (see Records/2026-07-09 — text_overlap over-recalls at k=5).
TOP_K_RULES = 3
OVERWRITE_OUTPUTS = False

# === image budget: capped at 1080p (Full HD) for VLM input ===================
# Longest edge <= 1920 and total pixels <= 1920x1080. Lower than the old 1568^2
# pixel budget; keeps the image-direct agent's per-turn image cost bounded.
IMAGE_LONGEST_EDGE = 1920
IMAGE_MAX_PIXELS = 1920 * 1080
IMAGE_MIN_PIXELS = 448 * 448
