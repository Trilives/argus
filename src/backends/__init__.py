"""Model backends for the CS line.

``get_backend`` resolves a backend name to a configured instance. The same
:class:`~backends.base.ChatBackend` interface covers both, so the pipeline and
the grep-agent retriever stay backend-agnostic.
"""

from __future__ import annotations

import config
from backends.base import ChatBackend, ChatResponse, ToolCall

__all__ = ["ChatBackend", "ChatResponse", "ToolCall", "get_backend"]


def get_backend(name: str, *, model: str | None = None,
                base_url: str | None = None) -> ChatBackend:
    """Return a backend instance.

    ``model`` overrides the configured model (the OpenAI endpoint serves many);
    it is ignored by the local backend, which is pinned to ``LOCAL_MODEL_DIR``.
    ``base_url`` points the OpenAI backend at an alternate server (cross-model
    robustness: a second model served on another host); ignored by local.
    """
    if name == "local_vllm":
        from backends.local_vllm import LocalVLLMBackend

        return LocalVLLMBackend()
    if name == "openai_api":
        from backends.openai_api import OpenAIBackend

        return OpenAIBackend(model=model or config.OPENAI_TEXT_MODEL, base_url=base_url)
    raise ValueError(f"Unknown backend: {name!r} (expected 'local_vllm' or 'openai_api')")
