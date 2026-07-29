# Models

The framework is model-agnostic: it drives any OpenAI-compatible endpoint. Model **weights are not
distributed here** — pull them from their public model cards. A single multimodal model can serve every
stage (image facts, the retrieval agent, and the judgement).

## Evaluated models

| Model | Role in the paper | HF id |
|---|---|---|
| Qwen3.5-9B | Vision + judge; statistically indistinguishable from the 35B judge on oracle judgement (delta GV-F1 0.010, 95% CI [-0.004, 0.025]) | `Qwen/Qwen3.5-9B` |
| Qwen3.6-35B-A3B-FP8 | Main-run vision model and judge | `Qwen/Qwen3.6-35B-A3B-FP8` |
| gemma-4-12B-it | Cross-model judge (robustness) | `google/gemma-4-12b-it` |
| SigLIP-2 (base/224) | R2 image-embedding retriever | `google/siglip2-base-patch16-224` |

The retrieval agents (R3/R4) require **tool calling**; the judge and fixed retrievers do not.

## Serving (local)

`run_demo.py` starts vLLM for you, but for reference the serve command for the default model — taken
from the Qwen3.5-9B model card — is:

```bash
vllm serve Qwen/Qwen3.5-9B --port 8000 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --max-model-len 32768
```

The tool-call parser is **model-specific** (`qwen3_coder` for Qwen3.5); override `ARGUS_VLLM_ARGS` for a
different model. Point the framework at the endpoint with `OPENAI_BASE_URL` / `OPENAI_API_KEY`
(`EMPTY` is fine for a local vLLM server).
