"""Token/latency accounting shared by the model backends.

``OpenAIBackend.complete`` previously discarded ``response.usage``, so no run
could report cost (docs/TODO.md "System and cost"). Every completed call now
lands in the module-level :data:`LEDGER`, keyed by ``(model, stage)``; runners
write ``LEDGER.snapshot()`` into their summaries so the big re-runs produce an
accuracy-cost table per cell as a side effect rather than needing a third pass.

The stage label is ambient (thread-local) rather than a ``complete`` parameter,
so the :class:`~backends.base.ChatBackend` protocol stays signature-stable:
pipeline stages wrap their model calls in ``with usage.stage("stage1_facts"):``
and any call made without a label books under ``"unlabeled"``.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

_local = threading.local()


@contextmanager
def stage(label: str) -> Generator[None]:
    """Set the ambient stage label for model calls made inside the block."""
    previous = getattr(_local, "stage", None)
    _local.stage = label
    try:
        yield
    finally:
        _local.stage = previous


def current_stage() -> str:
    return getattr(_local, "stage", None) or "unlabeled"


class UsageLedger:
    """Thread-safe accumulator of per-(model, stage) token counts and latency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cells: dict[tuple[str, str], dict[str, Any]] = {}

    def record(
        self,
        *,
        model: str,
        stage: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_s: float,
    ) -> None:
        with self._lock:
            cell = self._cells.setdefault(
                (model, stage),
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    # Calls whose endpoint reported no usage block; kept separate
                    # so a partial token total is never mistaken for a full one.
                    "calls_without_usage": 0,
                    "latency_s": 0.0,
                },
            )
            cell["calls"] += 1
            cell["latency_s"] += latency_s
            if prompt_tokens is None and completion_tokens is None:
                cell["calls_without_usage"] += 1
            else:
                cell["prompt_tokens"] += prompt_tokens or 0
                cell["completion_tokens"] += completion_tokens or 0

    def snapshot(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Nested copy ``{model: {stage: totals}}`` with latency rounded."""
        with self._lock:
            out: dict[str, dict[str, dict[str, Any]]] = {}
            for (model, stage_label), cell in sorted(self._cells.items()):
                out.setdefault(model, {})[stage_label] = {
                    **cell,
                    "latency_s": round(cell["latency_s"], 3),
                }
            return out

    def reset(self) -> None:
        with self._lock:
            self._cells.clear()


LEDGER = UsageLedger()
