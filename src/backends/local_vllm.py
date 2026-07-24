"""Local GPU backend.

Loads a model copied under ``CS/Model/`` with vLLM (or the Transformers path for
models vLLM cannot serve here) and runs single-image multimodal generation. This
is the backend for the VLM stages; tool-calling is not supported locally (the
grep-agent retriever runs on the OpenAI endpoint instead).

Heavy imports (torch / vLLM / Transformers) are deferred to first use so dry-run
scripts can import the package without touching the GPU.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

import config
from backends import usage as usage_ledger
from backends.base import ChatResponse

INTERNVL_EXTRA_SPECIAL_TOKENS = {
    "start_image_token": "<img>",
    "end_image_token": "</img>",
    "context_image_token": "<IMG_CONTEXT>",
    "video_token": "<|video_pad|>",
}

MINICPM_EXTRA_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "image_start_token": "<image>",
    "image_end_token": "</image>",
    "slice_start_token": "<slice>",
    "slice_end_token": "</slice>",
    "image_id_start_token": "<image_id>",
    "image_id_end_token": "</image_id>",
}

MINICPM_STOP_TOKEN_IDS = [248044, 248046]
MINICPM_DOWNSAMPLE_MODE = "16x"
MINICPM_MAX_SLICE_NUMS = 36


def _extra_special_tokens(model_name: str) -> dict[str, str] | None:
    if "internvl" in model_name:
        return INTERNVL_EXTRA_SPECIAL_TOKENS
    if "minicpm" in model_name:
        return MINICPM_EXTRA_SPECIAL_TOKENS
    return None


def _supports_qwen_pixel_budget(model_name: str) -> bool:
    return "qwen" in model_name and "minicpm" not in model_name


def _transformers_backend_name(model_name: str) -> str:
    if "minicpm-v-4.6" in model_name:
        return "transformers_minicpm"
    if "gemma-4" in model_name:
        return "transformers_gemma"
    return ""


def _with_image_payload(messages: list[dict[str, Any]], image: Image.Image) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        content = []
        for item in message["content"]:
            if isinstance(item, dict) and item.get("type") == "image":
                content.append({"type": "image", "image": image})
            else:
                content.append(item)
        converted.append({**message, "content": content})
    return converted


class LocalVLLMBackend:
    """vLLM / Transformers backend for a locally copied model directory."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.name = "local_vllm"
        self.model_dir = Path(model_dir or config.LOCAL_MODEL_DIR)
        self.model_name = self.model_dir.name.lower()
        self._model: Any = None
        self._processor: Any = None
        self._sampling_params: Any = None
        self._tf_backend = _transformers_backend_name(self.model_name)

    # -- loading ---------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Model not found: {self.model_dir}")

        from transformers import (
            AutoModelForImageTextToText,
            AutoModelForMultimodalLM,
            AutoProcessor,
        )
        from vllm import LLM, SamplingParams

        processor_kwargs: dict[str, Any] = {"trust_remote_code": True, "fix_mistral_regex": True}
        extra = _extra_special_tokens(self.model_name)
        if extra:
            processor_kwargs["extra_special_tokens"] = extra
        self._processor = AutoProcessor.from_pretrained(str(self.model_dir), **processor_kwargs)
        self._sampling_params = SamplingParams(
            temperature=config.TEMPERATURE,
            top_p=1.0,
            max_tokens=config.MAX_NEW_TOKENS,
            repetition_penalty=config.REPETITION_PENALTY,
            skip_special_tokens=False,
        )

        if self._tf_backend == "transformers_gemma":
            self._model = AutoModelForMultimodalLM.from_pretrained(
                str(self.model_dir), dtype="auto", device_map="auto", trust_remote_code=True
            )
            self._model.eval()
            return
        if self._tf_backend == "transformers_minicpm":
            self._model = AutoModelForImageTextToText.from_pretrained(
                str(self.model_dir), torch_dtype="auto", device_map="auto", trust_remote_code=True
            )
            self._model.eval()
            return

        llm_kwargs: dict[str, Any] = {
            "model": str(self.model_dir),
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.5,
            "max_model_len": 12288,
            "max_num_seqs": 1,
            "limit_mm_per_prompt": {"image": 1},
            "enforce_eager": False,
        }
        if _supports_qwen_pixel_budget(self.model_name):
            llm_kwargs["mm_processor_kwargs"] = {
                "min_pixels": config.IMAGE_MIN_PIXELS,
                "max_pixels": config.IMAGE_MAX_PIXELS,
            }
        self._model = LLM(**llm_kwargs)

    # -- inference -------------------------------------------------------------
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
        if tools:
            raise NotImplementedError(
                "local_vllm backend does not support tool calling; "
                "run the grep-agent retriever with AGENT_BACKEND='openai_api'."
            )
        if json_schema is not None:
            raise NotImplementedError(
                "constrained decoding (config.GUIDED_JSON) is wired for the "
                "openai_api backend only; serve the model behind vLLM instead."
            )
        self._ensure_loaded()
        started = time.perf_counter()
        text = self._generate(messages, image)
        usage_ledger.LEDGER.record(
            model=self.model_name,
            stage=usage_ledger.current_stage(),
            prompt_tokens=None,
            completion_tokens=None,
            latency_s=time.perf_counter() - started,
        )
        return ChatResponse(content=text, finish_reason="stop")

    def _generate(self, messages: list[dict[str, Any]], image: Image.Image | None) -> str:
        import torch

        if self._tf_backend == "transformers_minicpm":
            inputs = self._processor.apply_chat_template(
                _with_image_payload(messages, image) if image is not None else messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "downsample_mode": MINICPM_DOWNSAMPLE_MODE,
                    "max_slice_nums": MINICPM_MAX_SLICE_NUMS,
                },
                enable_thinking=config.ENABLE_THINKING,
            ).to(self._model.device)
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    downsample_mode=MINICPM_DOWNSAMPLE_MODE,
                    max_new_tokens=self._sampling_params.max_tokens,
                    do_sample=False,
                    repetition_penalty=config.REPETITION_PENALTY,
                    eos_token_id=MINICPM_STOP_TOKEN_IDS,
                )
            trimmed = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=False)
            ]
            return self._processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        if self._tf_backend == "transformers_gemma":
            inputs = self._processor.apply_chat_template(
                _with_image_payload(messages, image) if image is not None else messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=config.ENABLE_THINKING,
            ).to(self._model.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._sampling_params.max_tokens,
                    do_sample=False,
                    repetition_penalty=config.REPETITION_PENALTY,
                )
            response = self._processor.decode(outputs[0][input_len:], skip_special_tokens=False)
            if hasattr(self._processor, "parse_response"):
                parsed = self._processor.parse_response(response)
                if isinstance(parsed, dict):
                    return str(parsed.get("content") or parsed.get("thinking") or "").strip()
            return response

        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.ENABLE_THINKING,
        )
        request: dict[str, Any] = {"prompt": prompt}
        if image is not None:
            request["multi_modal_data"] = {"image": image}
        outputs = self._model.generate(
            request, sampling_params=self._sampling_params, use_tqdm=False
        )
        return outputs[0].outputs[0].text
