"""Prompt builders for CS evidence-chain experiments."""

from __future__ import annotations

import json
from typing import Any

import paths
from rules import load_rules


def load_prompt(path) -> str:
    return path.read_text(encoding="utf-8").strip()


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def image_fact_prompt() -> str:
    return load_prompt(paths.IMAGE_FACT_PROMPT_PATH)


def scene_fact_prompt() -> str:
    return load_prompt(paths.SCENE_FACT_PROMPT_PATH)


def evidence_chain_prompt() -> str:
    return load_prompt(paths.EVIDENCE_CHAIN_PROMPT_PATH)


def facts_rules_prompt() -> str:
    return load_prompt(paths.FACTS_RULES_PROMPT_PATH)


def image_rules_prompt() -> str:
    return load_prompt(paths.IMAGE_RULES_PROMPT_PATH)


def rule_evidence_prompt() -> str:
    return load_prompt(paths.RULE_EVIDENCE_PROMPT_PATH)


def rule_evidence_strict_prompt() -> str:
    return load_prompt(paths.RULE_EVIDENCE_STRICT_PROMPT_PATH)


def rule_evidence_batch_prompt() -> str:
    return load_prompt(paths.RULE_EVIDENCE_BATCH_PROMPT_PATH)


def judgement_prompt() -> str:
    return load_prompt(paths.JUDGEMENT_PROMPT_PATH)


def rule_aggregate_prompt() -> str:
    return load_prompt(paths.RULE_AGGREGATE_PROMPT_PATH)


def rule_aggregate_observation_prompt() -> str:
    return load_prompt(paths.RULE_AGGREGATE_OBSERVATION_PROMPT_PATH)


def rule_filter_prompt() -> str:
    return load_prompt(paths.RULE_FILTER_PROMPT_PATH)


def _image_text_messages(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]


def _text_messages(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


RULE_VOCABULARY_PLACEHOLDER = "<<RULE_VOCABULARY>>"


def rule_vocabulary() -> list[str]:
    """The domains the rule library covers, in the library's own words.

    Derived, never restated. The hand-written version of this list drifted from
    the library it described — it kept teaching "distribution box" after the
    rules had settled on "temporary power box", so the fact extractor named an
    object no rule contains and that rule became unreachable. ``subcategory`` is
    one of the fields retrieval searches, so every term here is reachable by
    construction and stays true as the library changes.
    """
    subcategories = {str(rule.get("subcategory") or "").strip().lower() for rule in load_rules()}
    return sorted(subcategories - {""})


# One shared Stage-1 schema/guardrails per FACT_MODE; only this one paragraph
# forks by RETRIEVAL_METHOD, since the two retrieval methods consume the fact
# list very differently: text_overlap scores it once with no retries (wants
# dense, canonical-vocabulary, exhaustive facts), while agent_grep gets its own
# multi-turn retry budget (wants a concise, prioritized list instead — an
# exhaustive dump just burns its iteration budget on noise).
_RETRIEVAL_TARGET_GUIDANCE = {
    "text_overlap": (
        "These facts feed a one-shot text-overlap retrieval with no second chance. "
        "Use the standard construction-safety term for each object, not a colloquial "
        f"paraphrase or synonym — the rule library covers {RULE_VOCABULARY_PLACEHOLDER}. "
        "List every object in the image that could relate to a rule — do not "
        "omit anything because its importance seems uncertain."
    ),
    "agent_grep": (
        "These facts will be used by an agent that can search the rule library over "
        "multiple turns on its own. List only representative scene objects actually "
        "present, ordered by importance, without repeating the same object. You need "
        "not enumerate every detail — the agent will search further for uncertain "
        "objects."
    ),
}


def build_fact_messages(*, mode: str = "generic", retrieval_method: str = "text_overlap") -> list[dict[str, Any]]:
    """Stage 1 fact extraction. ``scene`` mode yields retrieval-only scene facts
    with no verdict words; ``generic`` mode is the original broad fact pass.
    ``retrieval_method`` selects which retrieval-target guidance paragraph gets
    inserted (falls back to the ``text_overlap`` wording for unknown methods)."""
    prompt = scene_fact_prompt() if mode == "scene" else image_fact_prompt()
    guidance = _RETRIEVAL_TARGET_GUIDANCE.get(retrieval_method, _RETRIEVAL_TARGET_GUIDANCE["text_overlap"])
    prompt = prompt.replace("<<RETRIEVAL_TARGET_GUIDANCE>>", guidance)
    prompt = prompt.replace(RULE_VOCABULARY_PLACEHOLDER, ", ".join(rule_vocabulary()))
    return _image_text_messages(prompt)


def build_chain_messages(
    *,
    image_facts: list[str],
    candidate_rules: list[dict[str, Any]],
    unclear: list[str] = (),  # type: ignore[assignment]
) -> list[dict[str, Any]]:
    prompt = evidence_chain_prompt()
    prompt = prompt.replace("<<IMAGE_FACTS_JSON>>", compact_json(image_facts))
    prompt = prompt.replace("<<CANDIDATE_RULES_JSON>>", compact_json(candidate_rules))
    prompt = prompt.replace("<<UNCLEAR_JSON>>", compact_json(list(unclear)))
    return _image_text_messages(prompt)


def build_facts_rules_messages(
    *,
    image_facts: list[str],
    candidate_rules: list[dict[str, Any]],
    unclear: list[str] = (),  # type: ignore[assignment]
) -> list[dict[str, Any]]:
    """Protocol J1: Stage-1 facts + candidate rules, deliberately no image —
    the verdict must come from the pre-extracted facts alone."""
    prompt = facts_rules_prompt()
    prompt = prompt.replace("<<IMAGE_FACTS_JSON>>", compact_json(image_facts))
    prompt = prompt.replace("<<CANDIDATE_RULES_JSON>>", compact_json(candidate_rules))
    prompt = prompt.replace("<<UNCLEAR_JSON>>", compact_json(list(unclear)))
    return _text_messages(prompt)


def build_image_rules_messages(
    *,
    candidate_rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Protocol J2: original image + candidate rules, deliberately no Stage-1
    facts — the model observes the image in one pass."""
    prompt = image_rules_prompt()
    prompt = prompt.replace("<<CANDIDATE_RULES_JSON>>", compact_json(candidate_rules))
    return _image_text_messages(prompt)


def build_rule_evidence_messages(
    *,
    rule: dict[str, Any],
    scene_facts: list[str],
    unclear: list[str] = (),  # type: ignore[assignment]
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Stage 4: image + one rule -> per-checkpoint visual evidence (no verdict)."""
    prompt = rule_evidence_strict_prompt() if strict else rule_evidence_prompt()
    prompt = prompt.replace("<<RULE_JSON>>", compact_json(rule))
    prompt = prompt.replace("<<SCENE_FACTS_JSON>>", compact_json(scene_facts))
    prompt = prompt.replace("<<UNCLEAR_JSON>>", compact_json(list(unclear)))
    return _image_text_messages(prompt)


def build_rule_evidence_batch_messages(
    *,
    rules: list[dict[str, Any]],
    scene_facts: list[str],
    unclear: list[str] = (),  # type: ignore[assignment]
) -> list[dict[str, Any]]:
    """Stage 4 (batched): image + ALL candidate rules -> per-rule checkpoint
    evidence in one call (one image read instead of one per rule)."""
    prompt = rule_evidence_batch_prompt()
    prompt = prompt.replace("<<RULES_JSON>>", compact_json(rules))
    prompt = prompt.replace("<<SCENE_FACTS_JSON>>", compact_json(scene_facts))
    prompt = prompt.replace("<<UNCLEAR_JSON>>", compact_json(list(unclear)))
    return _image_text_messages(prompt)


def build_judgement_messages(
    *,
    rule_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Stage 5: text-only judgement over collected checkpoint evidence.

    Deliberately carries no image — the verdict must be grounded in the
    structured evidence from Stage 4, not free reasoning over the raw pixels.
    """
    prompt = judgement_prompt().replace("<<RULE_EVIDENCE_JSON>>", compact_json(rule_evidence))
    return _text_messages(prompt)


def build_rule_aggregate_messages(
    *,
    image_facts: list[str],
    candidate_rules: list[dict[str, Any]],
    unclear: list[str] = (),  # type: ignore[assignment]
) -> list[dict[str, Any]]:
    """Stage 2.5a: text-only — broad candidate rules -> one synthesized
    observation prompt covering all of their checkpoints (no image; the model
    only reorganizes rule metadata + facts already in hand)."""
    prompt = rule_aggregate_prompt()
    prompt = prompt.replace("<<IMAGE_FACTS_JSON>>", compact_json(image_facts))
    prompt = prompt.replace("<<CANDIDATE_RULES_JSON>>", compact_json(candidate_rules))
    prompt = prompt.replace("<<UNCLEAR_JSON>>", compact_json(list(unclear)))
    return _text_messages(prompt)


def build_rule_aggregate_observation_messages(*, aggregated_prompt: str) -> list[dict[str, Any]]:
    """Stage 2.5b: image + the model-generated aggregated prompt -> per-rule
    checkpoint observations (the "new round of features")."""
    prompt = rule_aggregate_observation_prompt().replace("<<AGGREGATED_PROMPT>>", aggregated_prompt)
    return _image_text_messages(prompt)


def build_rule_filter_messages(
    *,
    candidate_rules: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Stage 2.5c: text-only — observations decide which broad candidates are
    discarded before the judgement stage ever sees them."""
    prompt = rule_filter_prompt()
    prompt = prompt.replace("<<CANDIDATE_RULES_JSON>>", compact_json(candidate_rules))
    prompt = prompt.replace("<<OBSERVATIONS_JSON>>", compact_json(observations))
    return _text_messages(prompt)
