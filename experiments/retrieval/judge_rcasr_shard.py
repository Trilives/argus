#!/usr/bin/env python3
"""Run one disjoint shard of the frozen RCASR judge against a named server.

The frozen judge (`judge_rcasr.py`, pinned by `results/2026-09-22_rcasr_judge`) is
imported unchanged, and its snapshot is verified first. This runner only splits the
remaining pairs so that two servers of the same checkpoint (the shared endpoint and a
local copy of `Qwen3.8-27B-FP8`) never judge the same pair. Every row it writes carries
`server`, and `--agreement N` re-judges N pairs another server already judged. The result
goes to a side file, never the verdict cache, so the two serving stacks can be compared.

    uv run python experiments/retrieval/judge_rcasr_shard.py --population internal \
        --base-url http://localhost:8002/v1 --server local-4090 --shard 1/2 --workers 4
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import sys

CS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CS_ROOT / "src"))
sys.path.insert(0, str(CS_ROOT / "experiments" / "retrieval"))

import config  # noqa: E402
import judge_rcasr as jr  # noqa: E402
import rcasr_experiment as rx  # noqa: E402
from research_snapshot import read_json, verify_snapshot  # noqa: E402

SEED = 20260922


def shard_of(pair: tuple[str, str], count: int) -> int:
    """Deterministic, order-free shard assignment (stable across restarts)."""
    return random.Random(f"{SEED}:{pair[0]}:{pair[1]}").randrange(count)


def work_items(population: str) -> tuple[list, dict, dict]:
    return jr.internal_work() if population == "internal" else jr.public_work()


def run(args: argparse.Namespace, pairs: list, paths: dict, facts: dict, out: Path, tag: dict) -> None:
    from backends.openai_api import OpenAIBackend  # noqa: PLC0415
    from images import load_image  # noqa: PLC0415

    model = read_json(CS_ROOT / jr.CONFIG_PATH)["judge"]["model_id"]
    backend = OpenAIBackend(model, base_url=args.base_url, api_key="EMPTY")

    def work(pair):
        return pair, rx.judge_pair(backend, load_image(paths[pair[0]]), facts[pair[0]] or [], pair[1])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, future in enumerate(as_completed([pool.submit(work, p) for p in pairs]), 1):
            (image_id, rule_id), item = future.result()
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"image_id": image_id, "rule_id": rule_id, "model": model,
                                         "guided_json": True, "server": args.server, **tag, **item},
                                        ensure_ascii=False) + "\n")
            if n % 50 == 0 or n == len(pairs):
                print(f"{args.server}: {n}/{len(pairs)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--population", choices=("internal", "public"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server", required=True, help="label written into every row")
    parser.add_argument("--shard", default="0/1", help="k/n: judge pairs whose shard is k of n")
    parser.add_argument("--agreement", type=int, default=0,
                        help="re-judge N already-judged pairs into a side file instead")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    verify_snapshot(CS_ROOT, read_json(jr.OUT / "input_snapshot.json"))
    config.GUIDED_JSON = True
    pairs, paths, facts = work_items(args.population)
    cache = jr.OUT / f"{args.population}_pairs.jsonl"
    judged = [json.loads(l) for l in cache.read_text().splitlines()] if cache.exists() else []
    done = {(d["image_id"], d["rule_id"]) for d in judged}
    if args.agreement:
        sample = random.Random(SEED).sample(sorted(done), min(args.agreement, len(done)))
        run(args, sample, paths, facts, jr.OUT / f"{args.population}_server_agreement.jsonl", {"agreement": True})
        return
    k, n = map(int, args.shard.split("/"))
    todo = [p for p in pairs if p not in done and shard_of(p, n) == k]
    run(args, todo, paths, facts, cache, {})


if __name__ == "__main__":
    main()
