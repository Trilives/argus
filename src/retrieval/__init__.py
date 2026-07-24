"""Rule retrieval methods for the CS line.

``get_retriever`` resolves a method name to a configured :class:`Retriever`. The
two methods are compared head-to-head in the paper:

- ``text_overlap`` — dependency-free RAG baseline over the prebuilt rule index.
- ``agent_grep``   — a tool-calling LLM that greps the rule library itself.
"""

from __future__ import annotations

from retrieval.base import RetrievedRule, Retriever, chain_sample_query, facts_query

__all__ = [
    "RetrievedRule",
    "Retriever",
    "build_siglip_retriever",
    "build_visual_agent_retriever",
    "chain_sample_query",
    "facts_query",
    "get_retriever",
]


def get_retriever(method: str) -> Retriever:
    if method == "text_overlap":
        import paths
        from io_utils import load_json
        from retrieval.text_overlap import TextOverlapRetriever

        rule_index = load_json(paths.RULE_INDEX_PATH)
        if not isinstance(rule_index, list):
            raise ValueError(f"Expected list rule index at {paths.RULE_INDEX_PATH}")
        return TextOverlapRetriever(rule_index)
    if method == "bm25":
        import paths
        from io_utils import load_json
        from retrieval.bm25 import BM25Retriever

        rule_index = load_json(paths.RULE_INDEX_PATH)
        if not isinstance(rule_index, list):
            raise ValueError(f"Expected list rule index at {paths.RULE_INDEX_PATH}")
        return BM25Retriever(rule_index)
    if method == "agent_grep":
        from retrieval.agent_grep import AgentGrepRetriever

        return AgentGrepRetriever()
    if method == "agent_grep_visual":
        # Image-based: the pipeline routes it via retrieve_image (see
        # pipeline.retrieve_candidates); retrieve(query) raises by design.
        return build_visual_agent_retriever()
    raise ValueError(
        f"Unknown retrieval method: {method!r} "
        "(expected 'text_overlap', 'bm25', 'agent_grep', or 'agent_grep_visual')"
    )


def build_siglip_retriever(variant: str = "224", *, text_field: str | None = None):
    """Build a SigLIP-2 image->rule retriever for the given resolution variant.

    Image-based, so it is kept out of ``get_retriever`` (which returns text-query
    ``Retriever``s); call ``retrieve_image(image, top_k=...)`` on the result.
    """
    import config
    from rules import load_rule_units
    from retrieval.siglip import SiglipRetriever

    if variant not in config.SIGLIP_MODELS:
        raise ValueError(f"Unknown SigLIP variant {variant!r}; expected one of {list(config.SIGLIP_MODELS)}")
    return SiglipRetriever(
        config.SIGLIP_MODELS[variant],
        load_rule_units(),
        variant=variant,
        text_field=text_field or config.SIGLIP_TEXT_FIELD,
    )


def build_visual_agent_retriever():
    """Build the image-direct agentic-grep retriever.

    Image-based like SigLIP, so it is kept out of ``get_retriever``; call
    ``retrieve_image(image, top_k=..., facts=...)`` on the result.
    """
    from retrieval.agent_grep_visual import AgentGrepVisualRetriever

    return AgentGrepVisualRetriever()
