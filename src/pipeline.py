"""End-to-end CS evidence-chain pipeline.

One orchestrated flow, parameterised by retrieval method and model backend:

    image -> image facts (VLM) -> rule retrieval -> candidate rules
          -> evidence chain (VLM, checkpoint-by-checkpoint) -> schema validation

Two drivers share the per-item stages:

- :func:`run_on_samples` — runs over the curated ``evidence_chain_samples`` whose
  image facts and gold ``matched_rule`` are human-authored. Retrieval is scored
  head-to-head against the gold rule; fact extraction is skipped (facts given).
  Runs offline with the ``text_overlap`` method; the chain stage and the
  ``agent_grep`` method additionally need a backend.
- :func:`run_on_split` — the full pipeline over a dataset split: the VLM extracts
  facts from each image, then retrieval + chain generation. Requires a backend.

Backends and the retriever are injected, so the entry script
(``experiments/pipeline/run_pipeline.py``) owns all config wiring and the
``RUN_MODEL`` guard.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import config
import paths
import schemas
from backends import usage
from backends.base import ChatBackend
from dataset import ImageRecord
from images import load_image
from io_utils import extract_json_object, load_json
from prompts import (
    build_chain_messages,
    build_fact_messages,
    build_facts_rules_messages,
    build_image_rules_messages,
    build_judgement_messages,
    build_rule_evidence_batch_messages,
    build_rule_aggregate_messages,
    build_rule_aggregate_observation_messages,
    build_rule_evidence_messages,
    build_rule_filter_messages,
)
from retrieval import RetrievedRule, Retriever, chain_sample_query, facts_query
from rules import load_rule_units, rule_by_id
from validation import validate_chain_record


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- candidate rule payload ---------------------------------------------------
_CANDIDATE_FIELDS = (
    "rule_id",
    "major_category",
    "subcategory",
    "rule_name",
    "source_level",
    "source_quote",
    "decision_scope",
    "visual_checkpoints",
    "visual_screening_rule",
    "non_visual_fields",
    "review_prompt",
    "rectification_advice",
)


def candidate_rule_payload(rule_ids: list[str]) -> list[dict[str, Any]]:
    """Build the compact candidate-rule list handed to the chain prompt."""
    rules = rule_by_id(load_rule_units())
    payload = []
    for rule_id in rule_ids:
        rule = rules.get(rule_id)
        if rule is None:
            continue
        payload.append({field: rule.get(field) for field in _CANDIDATE_FIELDS})
    return payload


def _retrieval_payload(retrieved: list[RetrievedRule]) -> list[dict[str, Any]]:
    return [
        {"rank": item.rank, "rule_id": item.rule_id, "score": round(item.score, 6)}
        for item in retrieved
    ]


# --- rule refinement: broad retrieve -> aggregate prompt -> observe -> filter --
# Optional layer between retrieval and judgement (``config.RULE_REFINE_MODE``).
# The baseline hands the retriever's top-k straight to judgement, so only the
# fact-mediated query — never the image — shapes which rules survive retrieval
# (the "fact-mediated retrieval" property, see docs/ARCHITECTURE.md). This layer
# intentionally breaks that:
# it retrieves a much larger ``broad_top_k`` set, has the model synthesize one
# consolidated observation prompt over their checkpoints (Stage 2.5a), re-reads
# the image against that prompt for per-rule evidence (Stage 2.5b, the "new
# round of features"), then discards candidates unsupported by that evidence
# (Stage 2.5c) before the (expensive) judgement stage ever sees them.
def synthesize_rule_aggregate_prompt(
    backend: ChatBackend,
    image_facts: list[str],
    candidate_rules: list[dict[str, Any]],
    *,
    unclear: list[str] = (),  # type: ignore[assignment]
) -> tuple[str | None, str | None]:
    """Stage 2.5a: text-only pass merging every broad candidate's checkpoints
    into one aggregated observation prompt for Stage 2.5b."""
    messages = build_rule_aggregate_messages(
        image_facts=image_facts, candidate_rules=candidate_rules, unclear=list(unclear)
    )
    with usage.stage("stage2.5a_aggregate"):
        response = backend.complete(
            messages, image=None, max_tokens=config.MAX_NEW_TOKENS, json_schema=schemas.guided("aggregate")
        )
    parsed, error = extract_json_object(response.content or "")
    prompt_text = parsed.get("aggregated_prompt") if isinstance(parsed, dict) else None
    return (str(prompt_text) if prompt_text else None), error


def extract_aggregate_observations(
    backend: ChatBackend, image: Any, aggregated_prompt: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Stage 2.5b: the model-generated prompt + the image -> per-rule
    checkpoint observations."""
    messages = build_rule_aggregate_observation_messages(aggregated_prompt=aggregated_prompt)
    with usage.stage("stage2.5b_observe"):
        response = backend.complete(
            messages, image=image, max_tokens=config.MAX_NEW_TOKENS, json_schema=schemas.guided("observations")
        )
    parsed, error = extract_json_object(response.content or "")
    observations = parsed.get("observations", []) if isinstance(parsed, dict) else []
    return (observations if isinstance(observations, list) else []), error


def filter_candidate_rules(
    backend: ChatBackend, candidate_rules: list[dict[str, Any]], observations: list[dict[str, Any]]
) -> tuple[list[str], str | None]:
    """Stage 2.5c: text-only pass deciding which broad candidates survive."""
    messages = build_rule_filter_messages(candidate_rules=candidate_rules, observations=observations)
    with usage.stage("stage2.5c_filter"):
        response = backend.complete(
            messages, image=None, max_tokens=config.MAX_NEW_TOKENS, json_schema=schemas.guided("filter")
        )
    parsed, error = extract_json_object(response.content or "")
    kept = parsed.get("kept_rule_ids", []) if isinstance(parsed, dict) else []
    return ([str(item) for item in kept] if isinstance(kept, list) else []), error


def refine_candidate_rules(
    backend: ChatBackend,
    image: Any,
    image_facts: list[str],
    broad_rule_ids: list[str],
    *,
    unclear: list[str] = (),  # type: ignore[assignment]
) -> dict[str, Any]:
    """Run Stage 2.5a/b/c over a broad candidate set.

    Falls back to the unfiltered broad set whenever a stage fails to parse, so
    refinement can only narrow the judgement input, never silently drop it to
    nothing on a model hiccup. ``fallback`` names why the broad set was kept —
    a parse failure is a model/decoding defect, an empty keep-set is a genuine
    "nothing survived" verdict — so cells can report the two rates separately
    instead of the previous indistinguishable ``kept_ids or broad``.
    """
    candidate_rules = evidence_rule_payload(broad_rule_ids)
    aggregated_prompt, prompt_error = synthesize_rule_aggregate_prompt(
        backend, image_facts, candidate_rules, unclear=list(unclear)
    )
    if not aggregated_prompt:
        return {
            "rule_ids": broad_rule_ids,
            "aggregated_prompt": None,
            "observations": [],
            "fallback": "aggregate_parse_error" if prompt_error else "aggregate_empty",
            "errors": {"aggregate": prompt_error, "observe": None, "filter": None},
        }
    observations, observe_error = extract_aggregate_observations(backend, image, aggregated_prompt)
    kept_ids, filter_error = filter_candidate_rules(backend, candidate_rules, observations)
    fallback = None
    if not kept_ids:
        fallback = "filter_parse_error" if filter_error else "filter_empty"
    return {
        "rule_ids": kept_ids or broad_rule_ids,
        "aggregated_prompt": aggregated_prompt,
        "observations": observations,
        "fallback": fallback,
        "errors": {"aggregate": prompt_error, "observe": observe_error, "filter": filter_error},
    }


def retrieve_candidates(
    retriever: Retriever,
    query: str,
    *,
    top_k: int,
    refine_mode: str,
    broad_top_k: int,
    vlm_backend: ChatBackend | None,
    image: Any,
    image_facts: list[str],
    unclear: list[str] = (),  # type: ignore[assignment]
) -> tuple[list[RetrievedRule], dict[str, Any]]:
    """Stage 2 (+ optional Stage 2.5): retrieve candidate rules.

    ``refine_mode="none"`` is the baseline: one ``retriever.retrieve`` call at
    ``top_k``. ``"aggregate_filter"`` first retrieves a much larger
    ``broad_top_k`` set, then narrows it via :func:`refine_candidate_rules` —
    the one path where the image, not just the fact-mediated query, shapes
    which rules reach judgement.
    """
    # Image-based retrievers (agent_grep_visual) read the photo directly instead
    # of the fact-mediated text query; refine is a text-query path, so skip it.
    # Protocol R4 (EXPERIMENT_PROTOCOL.md §3 RQ1): no Stage-1 facts hint — the
    # facts= kwarg stays available only for a separately-labeled sensitivity arm.
    if hasattr(retriever, "retrieve_image"):
        retrieved = retriever.retrieve_image(image, top_k=top_k)  # type: ignore[attr-defined]
        extras = {}
        last = getattr(retriever, "last_result", None)
        if last is not None:
            extras["agent_status"] = {
                "finished": last.finished,
                "iterations": last.iterations,
                "error": last.error,
                "submitted_rule_ids": last.rule_ids,
            }
        return retrieved, extras

    if refine_mode == "none":
        return retriever.retrieve(query, top_k=top_k), {}
    if vlm_backend is None:
        raise ValueError("RULE_REFINE_MODE != 'none' requires a VLM backend (RUN_MODEL=True)")
    broad = retriever.retrieve(query, top_k=broad_top_k)
    refined = refine_candidate_rules(
        vlm_backend, image, image_facts, [item.rule_id for item in broad], unclear=list(unclear)
    )
    kept = set(refined["rule_ids"])
    retrieved = [item for item in broad if item.rule_id in kept] or broad
    return retrieved, {"rule_refine": {key: value for key, value in refined.items() if key != "rule_ids"}}


# --- per-stage helpers --------------------------------------------------------
def _facts_from_parsed(parsed: dict[str, Any], mode: str) -> list[str]:
    """Flatten a parsed fact object into the retrieval-query fact list.

    ``scene`` mode returns visible objects + spatial relations (no verdict
    words); ``generic`` mode returns the original ``image_facts`` list.
    """
    if mode == "scene":
        objects = parsed.get("scene_objects") or []
        relations = parsed.get("spatial_relations") or []
        return [str(item) for item in list(objects) + list(relations)]
    return [str(fact) for fact in parsed.get("image_facts", [])]


def _unclear_from_parsed(parsed: dict[str, Any]) -> list[str]:
    """Non-visual/unclear gaps Stage 1 flagged, in either fact mode.

    Threaded into judgement-stage prompts (chain / decoupled / rule_refine) so
    they default to ``need_review`` for what Stage 1 already couldn't confirm,
    instead of re-deriving (or inventing) an answer from scratch.
    """
    return [str(item) for item in (parsed.get("unclear_or_missing") or [])]


def extract_facts(
    backend: ChatBackend, image: Any, *, mode: str = "generic", retrieval_method: str = "text_overlap"
) -> tuple[list[str], dict[str, Any] | None, str | None]:
    """Stage 1: VLM extracts visible facts from one image.

    ``mode="scene"`` uses the retrieval-oriented scene prompt so facts stay
    objective and don't pre-bias the later judgement toward violations.
    ``retrieval_method`` swaps in a retrieval-target-specific instruction
    (see :func:`prompts.build_fact_messages`): ``text_overlap`` is a one-shot
    lexical scorer with no retries, so it wants dense, canonical-vocabulary,
    exhaustive facts; ``agent_grep`` gets its own retry budget, so it wants a
    concise, prioritized list instead of an exhaustive dump.
    """
    messages = build_fact_messages(mode=mode, retrieval_method=retrieval_method)
    with usage.stage("stage1_facts"):
        response = backend.complete(
            messages,
            image=image,
            max_tokens=config.MAX_NEW_TOKENS,
            json_schema=schemas.guided("facts_generic" if mode == "generic" else "facts_scene"),
        )
    parsed, error = extract_json_object(response.content or "")
    facts = _facts_from_parsed(parsed, mode) if parsed else []
    return facts, parsed, error


def generate_chain(
    backend: ChatBackend,
    image: Any,
    image_facts: list[str],
    candidate_rules: list[dict[str, Any]],
    *,
    unclear: list[str] = (),  # type: ignore[assignment]
) -> tuple[dict[str, Any] | None, str | None]:
    """Stage 3: VLM fills the evidence chain over the retrieved candidate rules.

    ``unclear`` is Stage 1's ``unclear_or_missing`` list — non-visual gaps it
    already flagged, so this pass defaults to ``need_review`` on them instead
    of re-deriving (or inventing) an answer.
    """
    messages = build_chain_messages(image_facts=image_facts, candidate_rules=candidate_rules, unclear=list(unclear))
    with usage.stage("stage3_chain"):
        response = backend.complete(
            messages, image=image, max_tokens=config.MAX_NEW_TOKENS, json_schema=schemas.guided("verdict")
        )
    return extract_json_object(response.content or "")


def judge_facts_rules(
    backend: ChatBackend,
    image_facts: list[str],
    candidate_rules: list[dict[str, Any]],
    *,
    unclear: list[str] = (),  # type: ignore[assignment]
) -> tuple[dict[str, Any] | None, str | None]:
    """Protocol J1: one-pass judgement from Stage-1 facts + rules, image=None.

    Same output schema as :func:`generate_chain`; the clean facts-only split of
    the legacy chain hybrid (EXPERIMENT_PROTOCOL.md §3 RQ2).
    """
    messages = build_facts_rules_messages(
        image_facts=image_facts, candidate_rules=candidate_rules, unclear=list(unclear)
    )
    with usage.stage("j1_facts_rules"):
        response = backend.complete(
            messages, image=None, max_tokens=config.MAX_NEW_TOKENS, json_schema=schemas.guided("verdict")
        )
    return extract_json_object(response.content or "")


def judge_image_rules(
    backend: ChatBackend,
    image: Any,
    candidate_rules: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Protocol J2: one-pass judgement from the original image + rules, no
    Stage-1 facts. Same output schema as :func:`generate_chain`; the clean
    image-only split of the legacy chain hybrid."""
    messages = build_image_rules_messages(candidate_rules=candidate_rules)
    with usage.stage("j2_image_rules"):
        response = backend.complete(
            messages, image=image, max_tokens=config.MAX_NEW_TOKENS, json_schema=schemas.guided("verdict")
        )
    return extract_json_object(response.content or "")


# Fields the VLM owns vs. fields the pipeline supplies. The VLM produces only the
# reasoning side of the schema; the dataset-side fields (ids, paths, raw facts)
# and the authoritative rule details are filled here, then the assembled record
# is validated against the one schema in data/chain/evidence_chain_schema.json.
def assemble_chain_record(
    chain: dict[str, Any] | None,
    *,
    image_id: str,
    image_path: Any,
    image_facts: list[str],
    sample_id: str | None = None,
    defect_category: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge pipeline-side fields onto the VLM chain output into a full record."""
    record: dict[str, Any] = dict(chain) if isinstance(chain, dict) else {}
    record["sample_id"] = sample_id or image_id
    record["image_id"] = image_id
    record["image_path"] = str(image_path)
    record["image_facts"] = image_facts

    rules = rule_by_id(load_rule_units())
    raw_matched = record.get("matched_rule")
    matched: dict[str, Any] = raw_matched if isinstance(raw_matched, dict) else {}
    matched_id = str(matched.get("rule_id") or "")
    rule = rules.get(matched_id) if matched_id else None
    if rule is not None:
        record["matched_rule"] = {
            "rule_id": rule["rule_id"],
            "rule_name": rule.get("rule_name"),
            "rule_text": rule.get("source_quote") or matched.get("rule_text") or "",
            "rule_source": rule.get("source_level"),
            "decision_scope": rule.get("decision_scope"),
        }
        record.setdefault(
            "defect_category",
            {"major_category": rule.get("major_category"), "subcategory": rule.get("subcategory")},
        )
    if defect_category is not None:
        record["defect_category"] = defect_category
    return record


# --- decoupled evidence -> judgement (Stage 4 + Stage 5) ----------------------
# Fields compiled into the per-rule evidence prompt. Unlike the chain payload,
# this carries the retrieval/exclusion keywords so the VLM can tell in vs out of
# scope, and drops rectification_advice (that belongs to the judgement stage).
_EVIDENCE_RULE_FIELDS = (
    "rule_id",
    "rule_name",
    "visual_checkpoints",
    "visual_screening_rule",
    "non_visual_fields",
    "decision_scope",
    "review_prompt",
    "positive_keywords",
    "exclusion_keywords",
)


def evidence_rule_payload(rule_ids: list[str]) -> list[dict[str, Any]]:
    """Compile the rule fields handed to the rule-conditioned evidence prompt."""
    rules = rule_by_id(load_rule_units())
    payload = []
    for rule_id in rule_ids:
        rule = rules.get(rule_id)
        if rule is None:
            continue
        payload.append({field: rule.get(field) for field in _EVIDENCE_RULE_FIELDS})
    return payload


def extract_rule_evidence(
    backend: ChatBackend,
    image: Any,
    scene_facts: list[str],
    rule_ids: list[str],
    *,
    unclear: list[str] = (),  # type: ignore[assignment]
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Stage 4: for each candidate rule, the VLM re-reads the image and reports
    per-checkpoint visual evidence (no verdict). One call per rule keeps each
    evidence pass focused on that rule's checkpoints.

    ``unclear`` is Stage 1's ``unclear_or_missing`` list, passed alongside
    ``scene_facts`` so a rule's non-visual checkpoints get flagged consistently
    with what Stage 1 already couldn't confirm, not re-derived per rule.
    ``strict`` selects the subject-strict prompt variant (J3-sym-strict arm):
    subject-gate checkpoints are judged against the rule's exact subject type.
    """
    evidence: list[dict[str, Any]] = []
    for rule in evidence_rule_payload(rule_ids):
        messages = build_rule_evidence_messages(
            rule=rule, scene_facts=scene_facts, unclear=list(unclear), strict=strict
        )
        with usage.stage("stage4_evidence"):
            response = backend.complete(
                messages, image=image, max_tokens=config.MAX_NEW_TOKENS, json_schema=schemas.guided("evidence")
            )
        parsed, error = extract_json_object(response.content or "")
        parsed = parsed if isinstance(parsed, dict) else {}
        evidence.append(
            {
                "rule_id": rule["rule_id"],
                "rule_name": rule.get("rule_name"),
                "visual_screening_rule": rule.get("visual_screening_rule"),
                "checkpoint_evidence": parsed.get("checkpoint_evidence", []),
                "missing_information": parsed.get("missing_information", []),
                "parse_error": error,
            }
        )
    return evidence


def extract_rule_evidence_batched(
    backend: ChatBackend,
    image: Any,
    scene_facts: list[str],
    rule_ids: list[str],
    *,
    unclear: list[str] = (),  # type: ignore[assignment]
) -> list[dict[str, Any]]:
    """Stage 4 (batched): one image read covering ALL candidate rules at once.

    Same output shape as :func:`extract_rule_evidence` (one entry per rule, in
    the input order) so Stage 5 and record assembly are unchanged. A rule absent
    from the model's output keeps empty evidence + a parse_error, mirroring the
    per-rule path's parse-failure handling.
    """
    rules = evidence_rule_payload(rule_ids)
    if not rules:
        return []
    messages = build_rule_evidence_batch_messages(rules=rules, scene_facts=scene_facts, unclear=list(unclear))
    # Output covers every candidate rule at once, so scale the budget with rule
    # count — a single-rule MAX_NEW_TOKENS would truncate the batch and fail to parse.
    batch_max_tokens = config.MAX_NEW_TOKENS * len(rules)
    with usage.stage("stage4_evidence"):
        response = backend.complete(
            messages, image=image, max_tokens=batch_max_tokens, json_schema=schemas.guided("evidence_batch")
        )
    parsed, error = extract_json_object(response.content or "")
    parsed = parsed if isinstance(parsed, dict) else {}
    by_id: dict[str, dict[str, Any]] = {}
    for item in parsed.get("rules", []):
        if isinstance(item, dict) and item.get("rule_id"):
            by_id[str(item["rule_id"])] = item

    evidence: list[dict[str, Any]] = []
    for rule in rules:
        rule_id = rule["rule_id"]
        item = by_id.get(rule_id, {})
        evidence.append(
            {
                "rule_id": rule_id,
                "rule_name": rule.get("rule_name"),
                "visual_screening_rule": rule.get("visual_screening_rule"),
                "checkpoint_evidence": item.get("checkpoint_evidence", []),
                "missing_information": item.get("missing_information", []),
                "parse_error": error if not item else None,
            }
        )
    return evidence


def judge_from_evidence(
    backend: ChatBackend, rule_evidence: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str | None]:
    """Stage 5: text-only judgement over the collected checkpoint evidence."""
    judge_input = [
        {
            "rule_id": item["rule_id"],
            "rule_name": item.get("rule_name"),
            "visual_screening_rule": item.get("visual_screening_rule"),
            "checkpoint_evidence": item.get("checkpoint_evidence", []),
            "missing_information": item.get("missing_information", []),
        }
        for item in rule_evidence
    ]
    messages = build_judgement_messages(rule_evidence=judge_input)
    with usage.stage("stage5_judgement"):
        response = backend.complete(
            messages, image=None, max_tokens=config.MAX_NEW_TOKENS, json_schema=schemas.guided("verdict")
        )
    return extract_json_object(response.content or "")


def _alignment_from_evidence(checkpoint_evidence: list[Any]) -> list[dict[str, Any]]:
    """Map Stage-4 checkpoint_evidence onto the schema's visual_checklist_alignment."""
    alignment: list[dict[str, Any]] = []
    for item in checkpoint_evidence:
        if not isinstance(item, dict):
            continue
        alignment.append(
            {
                "checkpoint": item.get("checkpoint", ""),
                "visual_evidence": item.get("visible_evidence", ""),
                "evidence_type": item.get("evidence_type", "not_visible"),
                "status": item.get("status", "need_review"),
                "confidence": item.get("confidence", 0.0),
                "note": "",
            }
        )
    return alignment


def assemble_decoupled_record(
    judgement: dict[str, Any] | None,
    rule_evidence: list[dict[str, Any]],
    *,
    image_id: str,
    image_path: Any,
    image_facts: list[str],
    sample_id: str | None = None,
    defect_category: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge the Stage-5 judgement with the chosen rule's Stage-4 evidence into a
    full evidence_chain record (so schema validation and eval stay unchanged)."""
    record: dict[str, Any] = dict(judgement) if isinstance(judgement, dict) else {}
    record["sample_id"] = sample_id or image_id
    record["image_id"] = image_id
    record["image_path"] = str(image_path)
    record["image_facts"] = image_facts

    rules = rule_by_id(load_rule_units())
    raw_matched = record.get("matched_rule")
    matched: dict[str, Any] = raw_matched if isinstance(raw_matched, dict) else {}
    matched_id = str(matched.get("rule_id") or "")
    rule = rules.get(matched_id) if matched_id else None

    chosen = next((item for item in rule_evidence if item.get("rule_id") == matched_id), None)
    record["visual_checklist_alignment"] = _alignment_from_evidence(
        chosen.get("checkpoint_evidence", []) if chosen else []
    )

    if rule is not None:
        record["matched_rule"] = {
            "rule_id": rule["rule_id"],
            "rule_name": rule.get("rule_name"),
            "rule_text": rule.get("source_quote") or matched.get("rule_text") or "",
            "rule_source": rule.get("source_level"),
            "decision_scope": rule.get("decision_scope"),
        }
        record.setdefault(
            "defect_category",
            {"major_category": rule.get("major_category"), "subcategory": rule.get("subcategory")},
        )
    if defect_category is not None:
        record["defect_category"] = defect_category
    return record


def run_judgement(
    vlm_backend: ChatBackend,
    image: Any,
    image_facts: list[str],
    rule_ids: list[str],
    *,
    judgement_mode: str,
    evidence_top_k: int,
    image_id: str,
    image_path: Any,
    sample_id: str | None = None,
    defect_category: dict[str, Any] | None = None,
    unclear: list[str] = (),  # type: ignore[assignment]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Dispatch the judgement stage; returns (assembled record, per-row extras).

    ``chain`` is the one-pass baseline; ``decoupled`` runs Stage 4 evidence
    extraction over the top-``evidence_top_k`` candidates then Stage 5 judgement.
    ``unclear`` is Stage 1's ``unclear_or_missing`` list (empty for the
    ``samples`` driver, which has no Stage-1 pass to derive it from).
    """
    if judgement_mode == "decoupled":
        extractor = extract_rule_evidence_batched if config.EVIDENCE_BATCHED else extract_rule_evidence
        rule_evidence = extractor(
            vlm_backend, image, image_facts, rule_ids[:evidence_top_k], unclear=list(unclear)
        )
        judgement, judge_error = judge_from_evidence(vlm_backend, rule_evidence)
        full = (
            assemble_decoupled_record(
                judgement,
                rule_evidence,
                image_id=image_id,
                image_path=image_path,
                image_facts=image_facts,
                sample_id=sample_id,
                defect_category=defect_category,
            )
            if judgement
            else None
        )
        return full, {"rule_evidence": rule_evidence, "chain_parse_error": judge_error}

    chain, chain_error = generate_chain(
        vlm_backend, image, image_facts, candidate_rule_payload(rule_ids), unclear=list(unclear)
    )
    full = (
        assemble_chain_record(
            chain,
            image_id=image_id,
            image_path=image_path,
            image_facts=image_facts,
            sample_id=sample_id,
            defect_category=defect_category,
        )
        if chain
        else None
    )
    return full, {"chain_parse_error": chain_error}


# --- drivers ------------------------------------------------------------------
def run_on_samples(
    retriever: Retriever,
    *,
    vlm_backend: ChatBackend | None = None,
    top_k: int = 5,
    judgement_mode: str = "chain",
    evidence_top_k: int = 3,
    refine_mode: str = "none",
    broad_top_k: int = 15,
) -> dict[str, Any]:
    """Run retrieval (+ optional judgement) over the curated samples.

    Facts are the human-authored gold facts, so this driver ignores ``FACT_MODE``
    but honours ``judgement_mode`` — with ``decoupled`` it becomes the
    gold-facts / retrieved-rule evidence-extraction ablation. ``refine_mode``
    (see :func:`retrieve_candidates`) requires ``vlm_backend`` since it re-reads
    the image; it cannot run in the fully offline (``vlm_backend=None``) mode.
    """
    samples = load_json(paths.EVIDENCE_CHAIN_SAMPLES_PATH)
    if not isinstance(samples, list):
        raise ValueError(f"Expected list in {paths.EVIDENCE_CHAIN_SAMPLES_PATH}")

    rows: list[dict[str, Any]] = []
    for sample in samples:
        image_facts = [str(fact) for fact in sample.get("image_facts", [])]
        image_path = paths.IMAGES_DIR / f"{sample['image_id']}.jpg"
        image = load_image(image_path) if (vlm_backend is not None or refine_mode != "none") else None
        retrieved, refine_extra = retrieve_candidates(
            retriever,
            chain_sample_query(sample),
            top_k=top_k,
            refine_mode=refine_mode,
            broad_top_k=broad_top_k,
            vlm_backend=vlm_backend,
            image=image,
            image_facts=image_facts,
        )
        rule_ids = [item.rule_id for item in retrieved]

        row: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "image_id": sample["image_id"],
            "target_rule_id": sample["matched_rule"]["rule_id"],
            "target_defect_category": sample["defect_category"],
            "retrieved": _retrieval_payload(retrieved),
            **refine_extra,
        }

        if vlm_backend is not None:
            full, extra = run_judgement(
                vlm_backend,
                image,
                image_facts,
                rule_ids,
                judgement_mode=judgement_mode,
                evidence_top_k=evidence_top_k,
                image_id=sample["image_id"],
                image_path=image_path,
                sample_id=sample.get("sample_id"),
                defect_category=sample.get("defect_category"),
            )
            row.update(extra)
            row["chain"] = full
            row["chain_validation_errors"] = validate_chain_record(full) if full else ["no chain parsed"]
        rows.append(row)

    return {
        "driver": "samples",
        "retrieval_method": retriever.method,
        "judgement_mode": judgement_mode,
        "rule_refine_mode": refine_mode,
        "vlm_backend": getattr(vlm_backend, "name", None),
        "created_at": utc_now(),
        "rows": rows,
    }


def run_on_split(
    records: list[ImageRecord],
    retriever: Retriever,
    vlm_backend: ChatBackend,
    *,
    top_k: int = 5,
    fact_mode: str = "generic",
    judgement_mode: str = "chain",
    evidence_top_k: int = 3,
    refine_mode: str = "none",
    broad_top_k: int = 15,
) -> dict[str, Any]:
    """Full pipeline over a dataset split: facts -> retrieve -> judgement.

    ``fact_mode`` picks Stage 1 (generic vs retrieval-oriented scene facts);
    ``judgement_mode`` picks the verdict path (one-pass chain vs decoupled
    evidence extraction + judgement); ``refine_mode`` picks whether retrieval
    also gets an image-conditioned aggregate-then-filter pass (see
    :func:`retrieve_candidates`).
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        image = load_image(record.image_path)
        image_facts, parsed_facts, facts_error = extract_facts(
            vlm_backend, image, mode=fact_mode, retrieval_method=retriever.method
        )
        unclear = _unclear_from_parsed(parsed_facts or {})
        retrieved, refine_extra = retrieve_candidates(
            retriever,
            facts_query(image_facts),
            top_k=top_k,
            refine_mode=refine_mode,
            broad_top_k=broad_top_k,
            vlm_backend=vlm_backend,
            image=image,
            image_facts=image_facts,
            unclear=unclear,
        )
        rule_ids = [item.rule_id for item in retrieved]
        full, extra = run_judgement(
            vlm_backend,
            image,
            image_facts,
            rule_ids,
            judgement_mode=judgement_mode,
            evidence_top_k=evidence_top_k,
            unclear=unclear,
            image_id=record.image_id,
            image_path=record.image_path,
        )
        rows.append(
            {
                "image_id": record.image_id,
                "label": record.label,
                "split": record.split,
                "image_facts": image_facts,
                "facts_parse_error": facts_error,
                "retrieved": _retrieval_payload(retrieved),
                **refine_extra,
                "chain": full,
                "chain_validation_errors": validate_chain_record(full) if full else ["no chain parsed"],
                **extra,
            }
        )
    return {
        "driver": "split",
        "retrieval_method": retriever.method,
        "fact_mode": fact_mode,
        "judgement_mode": judgement_mode,
        "rule_refine_mode": refine_mode,
        "vlm_backend": vlm_backend.name,
        "created_at": utc_now(),
        "rows": rows,
    }


def pipeline_manifest() -> dict[str, Any]:
    """Reproducibility record of the active configuration."""
    return {
        "retrieval_method": config.RETRIEVAL_METHOD,
        "fact_mode": config.FACT_MODE,
        "judgement_mode": config.JUDGEMENT_MODE,
        "evidence_top_k": config.EVIDENCE_TOP_K,
        "rule_refine_mode": config.RULE_REFINE_MODE,
        "broad_top_k_rules": config.BROAD_TOP_K_RULES,
        "vlm_backend": config.VLM_BACKEND,
        "agent_backend": config.AGENT_BACKEND,
        "agent_model": config.AGENT_MODEL,
        "local_model": config.LOCAL_MODEL_NAME,
        "openai_text_model": config.OPENAI_TEXT_MODEL,
        "openai_vlm_model": config.OPENAI_VLM_MODEL,
        "run_model": config.RUN_MODEL,
        "split_name": config.SPLIT_NAME,
        "limit_images": config.LIMIT_IMAGES,
        "top_k_rules": config.TOP_K_RULES,
        "generation": {
            "temperature": config.TEMPERATURE,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "enable_thinking": config.ENABLE_THINKING,
            "repetition_penalty": config.REPETITION_PENALTY,
            "guided_json": config.GUIDED_JSON,
        },
        "rule_index": str(paths.RULE_INDEX_PATH),
    }
