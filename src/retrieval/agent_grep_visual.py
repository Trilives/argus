"""Image-direct agentic grep retriever.

Same grep/read/submit loop as :mod:`retrieval.agent_grep`, but the agent looks at
the **image itself** every turn instead of a text fact list — attacking the
"fact-mediated" retrieval bottleneck (a rule the Stage-1 VLM never named in text
can still be recovered here). Needs a tool-capable *vision* backend (the endpoint
Qwen VLM); image is re-sent each turn, so it is heavier than text ``agent_grep``.
"""
from __future__ import annotations

from typing import Any

import config
from backends import get_backend
from backends.base import ChatBackend
from retrieval.agent_grep import AgentRetrievalResult, RuleCorpus, run_agent_loop
from retrieval.base import RetrievedRule

SYSTEM_PROMPT = (
    "You are the rule-retrieval agent of a construction-site compliance system. You "
    "see a construction-site image directly; find the rules in the library most "
    "relevant to the scene.\n\n"
    "Method:\n"
    "1. Observe the image and identify visible scene objects (opening, unprotected "
    "edge, scaffold, temporary power box, safety helmet, standing water, material "
    "stacking, etc.); when people are visible, also identify each person's "
    "action/task, location, and protective equipment (e.g. working at an edge, "
    "climbing the scaffold, no safety helmet, work at height without a harness).\n"
    "2. First cast a wide net: grep_rules with several different keywords and "
    "collect every plausibly applicable candidate with its rule_name. For "
    "worker-behavior rules, search with action words (climbing, edge work, safety "
    "harness), not only object nouns.\n"
    "3. Then filter: check each candidate's rule_name (use read_rule if unsure) "
    "against the image, and drop rules whose object or scene does not actually "
    "appear.\n"
    "4. Submit the survivors with submit_rules in descending relevance (best match "
    "first) — usually 1–3, at most 6; never pad.\n\n"
    "Note: search by scene/object/worker action, not by \"violation or not\" — a "
    "compliant scene should still recall the rules governing it. One image often "
    "involves several scenes; submit the rules for every visible scene."
)


def run_visual_agent_retrieval(
    image: Any, corpus: RuleCorpus, backend: ChatBackend, *, facts: list[str] | None = None
) -> AgentRetrievalResult:
    """Run the grep loop with the image in context (facts are an optional hint)."""
    hint = (
        f"\nReference text clues (possibly incomplete; the image prevails):\n{'; '.join(facts)}"
        if facts
        else ""
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Observe this construction-site image and retrieve the most relevant rules.{hint}"},
            ],
        },
    ]
    return run_agent_loop(messages, corpus, backend, image=image)


class AgentGrepVisualRetriever:
    """Image-based sibling of :class:`AgentGrepRetriever` (call ``retrieve_image``)."""

    method = "agent_grep_visual"

    def __init__(self, corpus: RuleCorpus | None = None, backend: ChatBackend | None = None) -> None:
        self.corpus = corpus or RuleCorpus.from_rules()
        # Vision backend (multimodal + tool calling): the endpoint VLM model.
        self.backend = backend or get_backend(
            config.VLM_BACKEND,
            model=config.OPENAI_VLM_MODEL if config.VLM_BACKEND == "openai_api" else None,
        )
        self.last_result: AgentRetrievalResult | None = None

    def retrieve_image(
        self, image: Any, *, top_k: int = 5, facts: list[str] | None = None
    ) -> list[RetrievedRule]:
        result = run_visual_agent_retrieval(image, self.corpus, self.backend, facts=facts)
        self.last_result = result
        return result.retrieved[:top_k] if top_k else result.retrieved

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedRule]:  # pragma: no cover
        raise NotImplementedError(
            "AgentGrepVisualRetriever is image-based; call retrieve_image(image, top_k=...) instead."
        )
