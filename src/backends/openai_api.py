"""OpenAI-compatible endpoint backend (vLLM / SGLang style servers).

Supports text, tool-calling (used by the grep-agent retriever), and — when the
endpoint serves a vision model — images via base64 ``data:`` URLs.
"""

from __future__ import annotations

import time
from typing import Any

from PIL import Image

import config
from backends import usage as usage_ledger
from backends.base import ChatResponse, ToolCall
from images import encode_image_data_url


def _inject_image(messages: list[dict[str, Any]], image: Image.Image) -> list[dict[str, Any]]:
    """Replace ``{"type": "image"}`` placeholders with an image_url part."""
    data_url = encode_image_data_url(image)
    converted: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message)
            continue
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
            else:
                parts.append(item)
        converted.append({**message, "content": parts})
    return converted


class OpenAIBackend:
    """Chat backend backed by an OpenAI-compatible HTTP endpoint."""

    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None) -> None:
        from openai import OpenAI

        self.name = "openai_api"
        self.model = model
        self._client = OpenAI(
            base_url=base_url or config.OPENAI_BASE_URL,
            api_key=api_key or config.OPENAI_API_KEY,
        )

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
    ) -> ChatResponse:
        if json_schema is not None and tools:
            raise ValueError("json_schema constrains plain generation; tool turns are already schema-bound")
        payload_messages = _inject_image(messages, image) if image is not None else messages
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": config.TEMPERATURE if temperature is None else temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        else:
            # Plain generation (VLM fact/chain stages). Two vLLM extras:
            # - repetition_penalty: avoid greedy repetition loops that break JSON.
            # - chat_template_kwargs.enable_thinking: Qwen3.6 otherwise spends the
            #   whole budget on `reasoning` and returns content=null (finish=length),
            #   so no JSON is produced. Tool calls keep the default (handled above).
            kwargs["extra_body"] = {
                "repetition_penalty": config.REPETITION_PENALTY,
                "chat_template_kwargs": {"enable_thinking": config.ENABLE_THINKING},
            }
            if json_schema is not None:
                # vLLM 0.25.1 constrains via OpenAI response_format json_schema;
                # its legacy extra_body guided_json is silently ignored (verified
                # live 2026-07-18 — the fenced output gave it away).
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "stage_output", "schema": json_schema},
                }

        started = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        latency_s = time.perf_counter() - started
        choice = response.choices[0]
        message = choice.message
        tool_calls = [
            ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments or "")
            for call in (message.tool_calls or [])
        ]
        reported = getattr(response, "usage", None)
        prompt_tokens = getattr(reported, "prompt_tokens", None) if reported else None
        completion_tokens = getattr(reported, "completion_tokens", None) if reported else None
        usage_ledger.LEDGER.record(
            model=self.model,
            stage=usage_ledger.current_stage(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency_s,
        )
        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=None
            if prompt_tokens is None and completion_tokens is None
            else {"prompt_tokens": prompt_tokens or 0, "completion_tokens": completion_tokens or 0},
            latency_s=latency_s,
        )
