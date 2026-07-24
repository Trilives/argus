"""Agentic grep retriever: the model searches the rule library itself.

The CS-line alternative to text-overlap RAG (:mod:`retrieval.text_overlap`).
Instead of scoring a precomputed index, a tool-calling LLM is handed three tools
over the canonical rule library and decides which rules to read and return:

- ``grep_rules(pattern)``    — case-insensitive regex over per-rule text blocks.
- ``read_rule(rule_id)``     — full structured detail for one rule.
- ``submit_rules(rule_ids)`` — terminal call; ranked most-relevant-first.

The loop runs on any tool-capable :class:`~backends.base.ChatBackend` (in
practice the OpenAI endpoint; the local backend does not support tools). Output
is a ranked ``list[RetrievedRule]`` so it is interchangeable with the baseline.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import config
from backends import get_backend, usage
from backends.base import ChatBackend
from retrieval.base import RetrievedRule
from rules import load_rules, rule_by_id

# Fields concatenated into each rule's greppable text block. With the English
# rule library (rules_en.json) the corpus is English, so the agent greps the same
# vocabulary the English fact extraction and rule metadata use.
_GREP_FIELDS = (
    "rule_id",
    "major_category",
    "subcategory",
    "rule_name",
    "visual_retrieval_text",
    "source_quote",
)


def _join_values(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(str(item) for item in value.values())
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value or "")


# Ablation: label which field a grep hit matched, and stop indexing
# exclusion_keywords as if they were positive evidence.
#
# The default (off) reproduces the runs already reported. With the flag on,
# `grep_rules` returns the matching field name, and terms that mean "this rule
# does NOT apply" no longer make the rule findable on that basis alone: today
# `grep("safety net installed")` returns R-OPN-001 solely because the phrase sits
# in its exclusion list -- the condition under which the rule is satisfied.
# Scope fields (`decision_scope`, `scenario_scope`) become greppable, so a rule's
# applicability boundary is visible at search time rather than only via read_rule.
GREP_FIX = os.environ.get("CS_GREP_FIX") == "1"

# Ablation: rank grep hits by match strength instead of corpus order.
#
# The default (off) reproduces the reported runs: `grep` walks the library in
# corpus order and keeps the first `max_hits`. At 42 rules a specific pattern
# rarely overflows the cap, so this is harmless today -- but it does not scale,
# and when it does overflow the survivors are an accident of rule ordering. With
# the flag on, every matching rule is scored (field weight x match count, see
# `_FIELD_WEIGHTS`) and the strongest survive truncation, so a pattern hitting a
# rule's name/keywords outranks one grazing its source clause or category.
AGENT_GREP_RANK = os.environ.get("CS_GREP_RANK") == "1"

# Field-strength weights for ranked grep. A hit in the rule name or the positive
# keywords is stronger relevance evidence than one in the source clause, scope,
# or category. Keys are the labels produced by `rule_field_blocks`.
_FIELD_WEIGHTS = {
    "rule_name": 5.0,
    "keywords": 4.0,
    "scene_text": 3.0,
    "checkpoint": 3.0,
    "rule_id": 2.0,
    "subcategory": 2.0,
    "category": 1.0,
    "source": 1.0,
    "scope": 1.0,
}
_DEFAULT_FIELD_WEIGHT = 1.0

_LABELLED_FIELDS = (
    ("rule_id", "rule_id"),
    ("major_category", "category"),
    ("subcategory", "subcategory"),
    ("rule_name", "rule_name"),
    ("visual_retrieval_text", "scene_text"),
    ("source_quote", "source"),
    ("scenario_scope", "scope"),
    ("decision_scope", "scope"),
)


def rule_field_blocks(rule: dict[str, Any]) -> list[tuple[str, str]]:
    """(field_label, text) pairs a labelled grep searches, exclusions omitted."""
    blocks = [(label, str(rule.get(name) or "")) for name, label in _LABELLED_FIELDS]
    blocks.append(("keywords", _join_values(rule.get("positive_keywords"))))
    blocks.append(("checkpoint", _join_values(rule.get("visual_checkpoints"))))
    return [(label, text) for label, text in blocks if text]


def rule_grep_block(rule: dict[str, Any]) -> str:
    """One newline-free searchable line per rule (id + scene/keyword fields)."""
    if GREP_FIX:
        return " | ".join(text for _label, text in rule_field_blocks(rule))
    parts = [str(rule.get(name) or "") for name in _GREP_FIELDS]
    parts.append(_join_values(rule.get("positive_keywords")))
    parts.append(_join_values(rule.get("exclusion_keywords")))
    parts.append(_join_values(rule.get("visual_checkpoints")))
    return " | ".join(part for part in parts if part)


def rule_detail(rule: dict[str, Any]) -> dict[str, Any]:
    """Compact structured view returned by ``read_rule``."""
    return {
        "rule_id": rule.get("rule_id"),
        "major_category": rule.get("major_category"),
        "subcategory": rule.get("subcategory"),
        "rule_name": rule.get("rule_name"),
        "decision_scope": rule.get("decision_scope"),
        "visual_checkpoints": rule.get("visual_checkpoints"),
        "positive_keywords": rule.get("positive_keywords"),
        "exclusion_keywords": rule.get("exclusion_keywords"),
        "source_quote": rule.get("source_quote"),
    }


@dataclass
class RuleCorpus:
    """In-memory greppable rule library backing the agent's tools."""

    rules: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self._by_id = rule_by_id(self.rules)
        self._blocks = [(rule["rule_id"], rule_grep_block(rule)) for rule in self.rules]
        self._fields = {rule["rule_id"]: rule_field_blocks(rule) for rule in self.rules}

    def _matched_field(self, rule_id: str, regex: re.Pattern[str]) -> str:
        """Which field the hit came from -- a snippet alone does not say.

        Without this the agent reads `... | cover fixed; sealed (blocked); ...`
        and cannot tell a rule_name match from a checkpoint or a scope boundary.
        """
        for label, text in self._fields.get(rule_id, []):
            if regex.search(text):
                return label
        return "unknown"

    def _hit_strength(self, rule_id: str, regex: re.Pattern[str]) -> float:
        """Match-strength score for ranked grep: sum of field weight x hit count.

        Scores over the positive field blocks only (exclusions are omitted from
        `_fields`), so a term meaning "rule does not apply" never lifts a rule's
        rank. A rule matched only on an exclusion (findable when GREP_FIX is off)
        scores 0 and sorts to the bottom.
        """
        score = 0.0
        for label, text in self._fields.get(rule_id, []):
            n = len(regex.findall(text))
            if n:
                score += _FIELD_WEIGHTS.get(label, _DEFAULT_FIELD_WEIGHT) * n
        return score

    @classmethod
    def from_rules(cls) -> "RuleCorpus":
        return cls(load_rules())

    def _format_hit(self, rule_id: str, block: str, match: re.Match[str],
                    regex: re.Pattern[str]) -> dict[str, str]:
        start = max(0, match.start() - 20)
        end = min(len(block), match.end() + 40)
        hit = {"rule_id": rule_id, "snippet": block[start:end].strip()}
        if GREP_FIX:
            hit["matched_field"] = self._matched_field(rule_id, regex)
        return hit

    def grep(self, pattern: str, *, max_hits: int = 12) -> list[dict[str, str]]:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return [{"error": f"invalid regex: {exc}"}]
        if AGENT_GREP_RANK:
            # Score every match, keep the strongest `max_hits`. Ties fall back to
            # corpus order (stable) so behaviour is deterministic.
            scored: list[tuple[float, int, str, str, re.Match[str]]] = []
            for idx, (rule_id, block) in enumerate(self._blocks):
                match = regex.search(block)
                if match is None:
                    continue
                scored.append((self._hit_strength(rule_id, regex), idx, rule_id, block, match))
            scored.sort(key=lambda t: (-t[0], t[1]))
            return [self._format_hit(rid, blk, m, regex) for _s, _i, rid, blk, m in scored[:max_hits]]
        hits: list[dict[str, str]] = []
        for rule_id, block in self._blocks:
            match = regex.search(block)
            if match is None:
                continue
            hits.append(self._format_hit(rule_id, block, match, regex))
            if len(hits) >= max_hits:
                break
        return hits

    def read(self, rule_id: str) -> dict[str, Any]:
        rule = self._by_id.get(rule_id)
        if rule is None:
            return {"error": f"unknown rule_id: {rule_id}"}
        return rule_detail(rule)

    def known(self, rule_id: str) -> bool:
        return rule_id in self._by_id


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "grep_rules",
            "description": (
                "Search the civilized-construction safety rule library with a "
                "case-insensitive regex. Returns matching rule_ids with snippets. "
                "Search with concrete scene nouns (opening, unprotected edge, safety "
                "helmet, temporary power box, ...), never with verdict words like "
                "\"violation\"/\"non-compliant\". The pattern matches literally, so "
                "prefer a short stem plus alternation: `scaffold` also finds "
                "\"scaffolding\" and \"scaffolds\", while `scaffolding` finds nothing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "regex, e.g. `opening|hole` or `safety helmet`",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_rule",
            "description": "Read one rule's full structure (checkpoints, decision scope, keywords, source clause).",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {"type": "string", "description": "rule id, e.g. R-OPN-001-horizontal-opening-protection"},
                },
                "required": ["rule_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_rules",
            "description": (
                "Submit the final result and end the task: rule_ids sorted by "
                "relevance, best match first. One image may involve several rules "
                "(e.g. an opening plus a missing helmet); submit every genuinely "
                "relevant rule — usually 1–3, at most 6 — and never pad with "
                "irrelevant ones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "rule_id list in descending relevance",
                    }
                },
                "required": ["rule_ids"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are the rule-retrieval agent of a construction-site compliance system. "
    "Given objective facts extracted from a site image, find the rules in the "
    "library most relevant to the scene.\n\n"
    "Method:\n"
    "1. Read the image facts and identify visible scene objects (opening, "
    "unprotected edge, scaffold, temporary power box, safety helmet, standing "
    "water, etc.) and worker actions (work at height, climbing, missing safety "
    "harness, etc.).\n"
    "2. First cast a wide net: grep_rules with several different keywords and "
    "collect every plausibly applicable candidate rule_id surfaced by the snippets.\n"
    "3. Then filter: inspect each candidate's rule_name with read_rule when needed "
    "against the fact list, and drop rules whose object or scene the facts do not "
    "support.\n"
    "4. Submit the survivors with submit_rules in descending relevance (best match "
    "first) — usually 1–3, at most 6; never pad.\n\n"
    "Note: search by scene/object, not by \"violation or not\" — a compliant scene "
    "(e.g. a properly covered opening) should still recall the rules governing it. "
    "Use specific search terms. One image often involves several scenes/objects; "
    "submit the rules for every visible scene, not just one."
)


@dataclass
class AgentRetrievalResult:
    rule_ids: list[str]
    retrieved: list[RetrievedRule]
    transcript: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    finished: bool = False
    error: str | None = None


def _normalize_rule_ids(raw: Any) -> list[str]:
    """Coerce a ``submit_rules`` argument into a clean list of rule_ids.

    The model sometimes hands back a (possibly truncated) JSON-encoded string
    instead of a real array; iterating that naively yields one char per rank.
    """
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw.split(",")
        raw = parsed
    if not isinstance(raw, list):
        return []
    cleaned = [str(item).strip().strip('[]"、 ') for item in raw]
    return [item for item in cleaned if item]


def _ranked(rule_ids: list[str], corpus: RuleCorpus) -> list[RetrievedRule]:
    retrieved: list[RetrievedRule] = []
    seen: set[str] = set()
    for rank, rule_id in enumerate(rule_ids, 1):
        if rule_id in seen or not corpus.known(rule_id):
            continue
        seen.add(rule_id)
        detail = corpus.read(rule_id)
        retrieved.append(
            RetrievedRule(
                rule_id=rule_id,
                score=float(len(rule_ids) - rank + 1),
                rank=len(retrieved) + 1,
                major_category=detail.get("major_category"),
                subcategory=detail.get("subcategory"),
                rule_name=detail.get("rule_name"),
            )
        )
    return retrieved


def _dispatch_tool(name: str, arguments: dict[str, Any], corpus: RuleCorpus) -> Any:
    if name == "grep_rules":
        return corpus.grep(str(arguments.get("pattern", "")), max_hits=config.AGENT_MAX_GREP_HITS)
    if name == "read_rule":
        return corpus.read(str(arguments.get("rule_id", "")))
    return {"error": f"unknown tool: {name}"}


def run_agent_retrieval(query: str, corpus: RuleCorpus, backend: ChatBackend) -> AgentRetrievalResult:
    """Run the grep-agent loop for one image-fact query on a tool-capable backend."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Image facts:\n{query}\n\nPlease retrieve the most relevant rules."},
    ]
    return run_agent_loop(messages, corpus, backend)


# Force the model to emit a submit_rules call (OpenAI-compatible tool_choice).
_SUBMIT_TOOL_CHOICE = {"type": "function", "function": {"name": "submit_rules"}}
# English nudge when a turn returns no tool call but still has budget left.
_RETRY_NUDGE = (
    "Please finish the retrieval with the tools: search with grep_rules, "
    "then call submit_rules to submit the result."
)
_SUBMIT_NUDGE = (
    "Stop searching and submit now: call submit_rules with the rule_ids you have "
    "found so far, ranked best match first."
)


def _append_assistant(messages: list[dict[str, Any]], response: Any) -> None:
    messages.append(
        {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in response.tool_calls
            ],
        }
    )


def _forced_submit(
    messages: list[dict[str, Any]],
    corpus: RuleCorpus,
    backend: ChatBackend,
    transcript: list[dict[str, Any]],
    *,
    iteration: int,
    reason: str,
    seen_hits: dict[str, int],
    image: Any = None,
) -> AgentRetrievalResult:
    """Best-effort terminal submit instead of returning an empty/error result.

    One extra turn with ``tool_choice`` pinned to ``submit_rules`` so the model
    ranks whatever it has; if that still yields nothing usable, fall back to the
    rule_ids grep surfaced so far (most-hit first). This turns budget exhaustion,
    stalls and truncation from "empty candidate set" into "anytime best-effort
    ranking" — the graceful degradation the protocol asks for.
    """
    messages.append({"role": "user", "content": _SUBMIT_NUDGE})
    try:
        with usage.stage("retrieval_agent_visual" if image is not None else "retrieval_agent"):
            response = backend.complete(
                messages,
                tools=TOOLS,
                tool_choice=_SUBMIT_TOOL_CHOICE,
                max_tokens=config.AGENT_MAX_TOKENS,
                temperature=config.AGENT_TEMPERATURE,
                image=image,
            )
    except Exception as exc:  # noqa: BLE001 - surface endpoint/network errors per query
        response = None
        transcript.append({"step": iteration, "forced_submit_error": str(exc)})

    rule_ids: list[str] = []
    if response is not None:
        _append_assistant(messages, response)
        for call in response.tool_calls:
            if call.name != "submit_rules":
                continue
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            rule_ids = _normalize_rule_ids(arguments.get("rule_ids"))
            break

    if not rule_ids:
        # Fall back to the grep-hit accumulation, most-frequently-hit first.
        rule_ids = [rid for rid, _ in sorted(seen_hits.items(), key=lambda kv: -kv[1]) if corpus.known(rid)]

    transcript.append({"step": iteration, "forced_submit": reason, "rule_ids": rule_ids})
    retrieved = _ranked(rule_ids, corpus)
    # Best-effort recovery can still come up empty (recovery turn failed and grep
    # surfaced nothing). That is a retrieval failure, not a finished retrieval:
    # `agent_failure_rate` keys off `finished`, so claiming success here would
    # under-report the very failure mode this path exists to catch.
    return AgentRetrievalResult(
        rule_ids=rule_ids,
        retrieved=retrieved,
        transcript=transcript,
        iterations=iteration,
        finished=bool(retrieved),
        error=None if retrieved else f"forced submit produced no known rule_id ({reason})",
    )


def run_agent_loop(
    messages: list[dict[str, Any]], corpus: RuleCorpus, backend: ChatBackend, *, image: Any = None
) -> AgentRetrievalResult:
    """Drive the grep tool-calling loop over ``messages`` until ``submit_rules``.

    ``image`` (optional) is passed to every ``complete`` call so an image-direct
    agent keeps the photo in context across turns; text-only callers leave it None.

    Termination is anytime: the loop tracks the cumulative set of grep-surfaced
    rule_ids and the patterns already run, and forces a best-effort submit (rather
    than returning empty) when (a) it reaches the final iteration, (b) it stalls —
    ``AGENT_STALL_LIMIT`` consecutive grep calls add no new rule_id or merely
    repeat a prior pattern, (c) a turn is truncated / emits no tool call, or
    (d) ``submit_rules`` arrives but carries no usable rule_id — truncated
    arguments or ids outside the corpus.

    Invariant: a returned result is never both ``finished=True`` and empty. An
    empty candidate set is a retrieval miss, so reporting it as finished would
    hide it from the per-cell failure rate.
    """
    transcript: list[dict[str, Any]] = []
    empty_turns = 0
    seen_hits: dict[str, int] = {}  # rule_id -> cumulative grep hits (fallback ranking)
    executed_patterns: set[str] = set()
    no_progress = 0  # consecutive grep calls with no new rule_id / repeated pattern

    for iteration in range(1, config.AGENT_MAX_ITERS + 1):
        # On the final allowed iteration, pin tool_choice to submit so the budget
        # ends on a ranking rather than an empty result.
        last_iteration = iteration == config.AGENT_MAX_ITERS
        try:
            with usage.stage("retrieval_agent_visual" if image is not None else "retrieval_agent"):
                response = backend.complete(
                    messages,
                    tools=TOOLS,
                    tool_choice=_SUBMIT_TOOL_CHOICE if last_iteration else "auto",
                    max_tokens=config.AGENT_MAX_TOKENS,
                    temperature=config.AGENT_TEMPERATURE,
                    image=image,
                )
        except Exception as exc:  # noqa: BLE001 - surface endpoint/network errors per query
            return AgentRetrievalResult([], [], transcript, iteration - 1, False, str(exc))

        _append_assistant(messages, response)

        if not response.tool_calls:
            empty_turns += 1
            truncated = response.finish_reason == "length"
            transcript.append(
                {"step": iteration, "assistant": response.content, "finish_reason": response.finish_reason}
            )
            # Truncation (finish_reason=length) or a second empty turn: recover with
            # a forced submit instead of failing to an empty candidate set.
            if truncated or empty_turns >= 2 or last_iteration:
                reason = "truncated" if truncated else ("no tool calls" if empty_turns >= 2 else "budget exhausted")
                return _forced_submit(
                    messages, corpus, backend, transcript,
                    iteration=iteration, reason=reason, seen_hits=seen_hits, image=image,
                )
            messages.append({"role": "user", "content": _RETRY_NUDGE})
            continue

        unusable_submit = False
        for call in response.tool_calls:
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            if call.name == "submit_rules":
                rule_ids = _normalize_rule_ids(arguments.get("rule_ids"))
                retrieved = _ranked(rule_ids, corpus)
                transcript.append({"step": iteration, "tool": call.name, "arguments": arguments})
                if retrieved:
                    return AgentRetrievalResult(
                        rule_ids=rule_ids,
                        retrieved=retrieved,
                        transcript=transcript,
                        iterations=iteration,
                        finished=True,
                    )
                # Submit carried nothing usable: finish_reason=length drops the
                # argument JSON tail (so `arguments` parses to {}), or every id is
                # unknown. Answering the call keeps the transcript protocol-valid
                # for the recovery turn appended after this loop; returning here
                # would score a retrieval miss as finished=True.
                unusable_submit = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            {"error": "no known rule_id parsed from rule_ids", "action": "resubmit"},
                            ensure_ascii=False,
                        ),
                    }
                )
                continue

            repeated = False
            if call.name == "grep_rules":
                pattern = str(arguments.get("pattern", ""))
                repeated = pattern in executed_patterns
                executed_patterns.add(pattern)

            result = _dispatch_tool(call.name, arguments, corpus)

            if call.name == "grep_rules":
                new_ids = 0
                if isinstance(result, list):
                    for hit in result:
                        rid = hit.get("rule_id") if isinstance(hit, dict) else None
                        if not rid:
                            continue
                        if rid not in seen_hits:
                            new_ids += 1
                        seen_hits[rid] = seen_hits.get(rid, 0) + 1
                # A repeated pattern or a search that surfaced nothing new is no
                # marginal information gain — count it toward the stall budget.
                no_progress = no_progress + 1 if (repeated or new_ids == 0) else 0

            transcript.append(
                {"step": iteration, "tool": call.name, "arguments": arguments, "result": result}
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})

        if unusable_submit:
            return _forced_submit(
                messages, corpus, backend, transcript,
                iteration=iteration, reason="unusable submit", seen_hits=seen_hits, image=image,
            )

        # Marginal-information-gain early stop: enough consecutive fruitless greps.
        if no_progress >= config.AGENT_STALL_LIMIT:
            return _forced_submit(
                messages, corpus, backend, transcript,
                iteration=iteration, reason="no new information", seen_hits=seen_hits, image=image,
            )

    # Unreachable in practice (last iteration forces submit), but keep a safety net.
    return _forced_submit(
        messages, corpus, backend, transcript,
        iteration=config.AGENT_MAX_ITERS, reason="max_iters", seen_hits=seen_hits, image=image,
    )


class AgentGrepRetriever:
    """Retriever wrapper so the agent loop is interchangeable with the baseline."""

    method = "agent_grep"

    def __init__(self, corpus: RuleCorpus | None = None, backend: ChatBackend | None = None) -> None:
        self.corpus = corpus or RuleCorpus.from_rules()
        self.backend = backend or get_backend(config.AGENT_BACKEND, model=config.AGENT_MODEL)
        self.last_result: AgentRetrievalResult | None = None

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedRule]:
        result = run_agent_retrieval(query, self.corpus, self.backend)
        self.last_result = result
        return result.retrieved[:top_k] if top_k else result.retrieved
