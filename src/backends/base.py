"""Common types for CS model backends.

A backend turns a list of chat messages into a :class:`ChatResponse`. Two
implementations exist — :mod:`backends.local_vllm` (GPU) and
:mod:`backends.openai_api` (remote endpoint) — so the same pipeline can
run any served model by switching ``config.VLM_BACKEND`` / ``config.AGENT_BACKEND``.

Messages follow the OpenAI content-list shape. An image is passed separately to
:meth:`ChatBackend.complete` and substituted into the first ``{"type": "image"}``
placeholder, so prompt builders stay backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as emitted by the model


@dataclass(frozen=True)
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    # Endpoint-reported token counts ({"prompt_tokens", "completion_tokens"})
    # and wall-clock latency; None when the backend/endpoint reports neither.
    # Every call is also accumulated in backends.usage.LEDGER by (model, stage).
    usage: dict[str, int] | None = None
    latency_s: float | None = None


class ChatBackend(Protocol):
    """Minimal chat interface shared by local and remote model backends."""

    name: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        image: Image.Image | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResponse: ...
