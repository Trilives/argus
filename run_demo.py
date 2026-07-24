#!/usr/bin/env python3
"""ARGus end-to-end demo runner.

Runs the full screening pipeline on the bundled sample images with a single
multimodal model (default **Qwen3.5-9B**, which handles the image stages, the
retrieval agent, and the judgement). Two interchangeable modes:

  * API mode        — point at an existing OpenAI-compatible endpoint.
  * Local-service   — start vLLM from local weights (or download by HF id).

    uv sync --group serving
    uv run python run_demo.py                      # local service (default)
    OPENAI_BASE_URL=http://host:8000/v1 uv run python run_demo.py   # API mode

Environment variables (defaults in brackets):

    ARGUS_MODE        auto | api | local            [auto: api if OPENAI_BASE_URL set, else local]
    ARGUS_MODEL       HF id served when no local dir found   [Qwen/Qwen3.5-9B]
    ARGUS_MODEL_PATH  explicit local weights dir            [auto-detect Model/Qwen3.5-9B]
    ARGUS_SERVED_NAME model id the backend sends            [Qwen3.5-9B]
    ARGUS_PORT        vLLM port                             [8000]
    ARGUS_VLLM_ARGS   serve flags (from the model README)
                      [--tensor-parallel-size 1 --max-model-len 32768
                       --reasoning-parser qwen3 --enable-auto-tool-choice
                       --tool-call-parser qwen3_coder]
    ARGUS_RETRIEVAL   text_overlap | agent_grep | agent_grep_visual  [agent_grep_visual]
    ARGUS_JUDGEMENT   decoupled (J3) | chain        [decoupled]
    ARGUS_NUM_IMAGES  sample images to screen       [3]

    # API mode also reads:
    OPENAI_BASE_URL   the endpoint (e.g. http://host:8000/v1)
    OPENAI_API_KEY    key, or "EMPTY" for a local vLLM/SGLang server

Serve flags are taken from the Qwen3.5-9B model card (tool-call-parser
qwen3_coder). Override ARGUS_VLLM_ARGS for a different model.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from shutil import which
from typing import Any, Callable, NoReturn

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
SAMPLE_DIR = ROOT / "examples" / "sample_images"
OUT_PATH = ROOT / "results" / "demo" / "demo_result.json"

MODE = os.environ.get("ARGUS_MODE", "auto")
MODEL_HF_ID = os.environ.get("ARGUS_MODEL", "Qwen/Qwen3.5-9B")
MODEL_PATH_ENV = os.environ.get("ARGUS_MODEL_PATH")
SERVED_NAME = os.environ.get("ARGUS_SERVED_NAME", "Qwen3.5-9B")
PORT = int(os.environ.get("ARGUS_PORT", "8000"))
DEFAULT_VLLM_ARGS = (
    "--tensor-parallel-size 1 --max-model-len 32768 "
    "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
)
VLLM_ARGS = os.environ.get("ARGUS_VLLM_ARGS", DEFAULT_VLLM_ARGS)
RETRIEVAL = os.environ.get("ARGUS_RETRIEVAL", "agent_grep_visual")
JUDGEMENT = os.environ.get("ARGUS_JUDGEMENT", "decoupled")
NUM_IMAGES = int(os.environ.get("ARGUS_NUM_IMAGES", "3"))
READY_TIMEOUT_S = int(os.environ.get("ARGUS_READY_TIMEOUT_S", "1800"))


def fail(message: str) -> NoReturn:
    print(f"\n[demo] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_gpu() -> None:
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError:
        fail("PyTorch/vLLM not installed. Run: uv sync --group serving")
    if not torch.cuda.is_available():  # type: ignore[union-attr]
        print("[demo] WARNING: no CUDA GPU detected — local serving will fail or be very slow.")


def local_model_dir() -> Path | None:
    """Find local Qwen3.5-9B weights, referencing the repo's Model/ tree if present."""
    if MODEL_PATH_ENV:
        p = Path(MODEL_PATH_ENV)
        return p if p.is_dir() else None
    name = SERVED_NAME
    for base in (ROOT / "Model", ROOT.parent / "Model"):  # public bundle, then parent repo
        cand = base / name
        if cand.is_dir():
            return cand
    return None


def endpoint_ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def use_api() -> tuple[None, str, str]:
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        fail("API mode needs OPENAI_BASE_URL (e.g. http://host:8000/v1).")
    if not endpoint_ready(base_url):
        fail(f"Endpoint {base_url} is not reachable.")
    model_id = os.environ.get("ARGUS_SERVED_NAME") or MODEL_HF_ID
    print(f"[demo] API mode: {base_url} (model={model_id})")
    return None, base_url, model_id


def start_local_service() -> tuple[subprocess.Popen[bytes], str, str]:
    if not which("vllm"):
        fail("`vllm` not found. Run: uv sync --group serving")
    target = local_model_dir()
    serve_arg = str(target) if target else MODEL_HF_ID
    if target:
        print(f"[demo] Local service from existing weights: {target}")
    else:
        print(f"[demo] No local weights found; vLLM will download {MODEL_HF_ID} from HuggingFace.")

    base_url = f"http://localhost:{PORT}/v1"
    cmd = ["vllm", "serve", serve_arg, "--port", str(PORT),
           "--served-model-name", SERVED_NAME, *shlex.split(VLLM_ARGS)]
    print(f"[demo] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)

    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            fail(f"vLLM exited early (code {proc.returncode}). Check ARGUS_VLLM_ARGS / GPU memory.")
        if endpoint_ready(base_url):
            print(f"[demo] Server ready at {base_url}")
            return proc, base_url, SERVED_NAME
        time.sleep(5)
    proc.terminate()
    fail(f"Server did not become ready within {READY_TIMEOUT_S}s.")


def acquire_backend() -> tuple[subprocess.Popen[bytes] | None, str, str]:
    mode = MODE
    if mode == "auto":
        mode = "api" if os.environ.get("OPENAI_BASE_URL") else "local"
    if mode == "api":
        return use_api()
    if mode == "local":
        check_gpu()
        return start_local_service()
    fail(f"Unknown ARGUS_MODE={MODE!r} (use auto|api|local).")


def sample_records(n: int, record_cls: Callable[..., Any]) -> list[Any]:
    images = sorted(SAMPLE_DIR.glob("*.jpg"))
    if not images:
        fail(f"No sample images in {SAMPLE_DIR}")
    return [
        record_cls(
            image_id=p.stem,
            image_path=p,
            label="",  # demo images carry no gold label
            source_folder="demo",
            original_name=p.name,
            split="demo",
        )
        for p in images[:n]
    ]


def run_pipeline(base_url: str, model_id: str) -> dict:
    # Point the framework at the endpoint BEFORE importing config-consuming
    # modules (config reads OPENAI_BASE_URL from the environment at import time).
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("CS_RULES_PATH", str(ROOT / "data" / "rules" / "rules_en.json"))
    sys.path.insert(0, str(SRC))

    import config  # noqa: PLC0415
    from backends import get_backend  # noqa: PLC0415
    from dataset import ImageRecord  # noqa: PLC0415
    from pipeline import pipeline_manifest, run_on_split  # noqa: PLC0415
    from retrieval import get_retriever  # noqa: PLC0415

    # One multimodal model serves every stage (facts, agent, judgement).
    config.OPENAI_VLM_MODEL = model_id
    config.OPENAI_TEXT_MODEL = model_id
    config.AGENT_MODEL = model_id

    retriever = get_retriever(RETRIEVAL)
    backend = get_backend("openai_api", model=model_id)
    records = sample_records(NUM_IMAGES, ImageRecord)
    print(f"[demo] Screening {len(records)} image(s): retrieval={RETRIEVAL}, judgement={JUDGEMENT} ...")

    result = run_on_split(
        records,
        retriever,
        backend,
        top_k=config.TOP_K_RULES,
        fact_mode=config.FACT_MODE,
        judgement_mode=JUDGEMENT,
        evidence_top_k=config.EVIDENCE_TOP_K,
        refine_mode=config.RULE_REFINE_MODE,
        broad_top_k=config.BROAD_TOP_K_RULES,
    )
    result["manifest"] = pipeline_manifest()
    result["demo"] = {"model": model_id, "retrieval": RETRIEVAL, "judgement": JUDGEMENT}
    return result


def print_summary(result: dict) -> None:
    print("\n" + "=" * 68 + "\nDemo results (auditable chains)\n" + "=" * 68)
    for row in result.get("rows", []):
        img = row.get("image_id", "?")
        chain = row.get("chain") or row.get("record") or {}
        verdicts = chain.get("clause_verdicts") or chain.get("verdicts") or row.get("verdicts") or []
        print(f"\n[{img}]")
        if verdicts:
            for v in verdicts:
                print(f"  - {v.get('rule_id', '?')}: {v.get('status', '?')}")
        else:
            print(f"  (raw row keys: {list(row)[:8]})")


def main() -> None:
    proc, base_url, model_id = acquire_backend()
    try:
        result = run_pipeline(base_url, model_id)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print_summary(result)
        print(f"\n[demo] Full result written to {OUT_PATH.relative_to(ROOT)}")
    finally:
        if proc is not None:
            print("[demo] Shutting down the vLLM server ...")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
