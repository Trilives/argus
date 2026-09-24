#!/usr/bin/env python3
"""The frozen per-pair judge for RCASR E2 (internal) and E5b (public); reads no gold.

J3-sym on the endpoint model named in ``rcasr_v1.json``, with constrained JSON decoding
(``config.GUIDED_JSON`` switched on here, for every arm alike). Without it the 27B judge
emitted malformed JSON on ~17% of pairs, and a parse failure falls through to the legacy
``compliant`` default — a silent bias toward misses. The first, unconstrained attempt is
kept in ``results/2026-09-22_rcasr_v2/judge_pairs.jsonl`` and never scored.

Usage::

    uv run python experiments/retrieval/judge_rcasr.py --prepare
    uv run python experiments/retrieval/judge_rcasr.py --judge internal --workers 32
    uv run python experiments/retrieval/judge_rcasr.py --judge public --workers 32
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

CS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CS_ROOT / "src"))
sys.path.insert(0, str(CS_ROOT / "experiments" / "retrieval"))

import config  # noqa: E402
import rcasr_experiment as rx  # noqa: E402
from research_snapshot import read_json, snapshot, verify_snapshot  # noqa: E402

OUT = CS_ROOT / "results/2026-09-22_rcasr_judge"
CONFIG_PATH = "data/rules/proposed/rcasr_v1.json"
INTERNAL_ROWS = "results/2026-09-22_rcasr_v2/rcasr_rows.jsonl"
PUBLIC_DIR = "results/2026-09-22_rcasr_public"
FILES = ("experiments/retrieval/judge_rcasr.py", "src/rcasr_experiment.py", "src/pipeline.py",
         "src/prompts.py", "src/schemas.py", "src/symbolic_judgement.py", "src/rules.py",
         "src/images.py", "src/backends/openai_api.py", "src/config.py", CONFIG_PATH,
         "Prompts_en/decoupled/rule_evidence_prompt.md", "data/rules/rules_en.json",
         "data/rule_assets/rule_units.json", "results/retrieval/gold_facts_generic_en.json",
         INTERNAL_ROWS)


def prepare() -> None:
    if (OUT / "input_snapshot.json").exists():
        raise FileExistsError("existing freeze cannot be overwritten")
    frozen = snapshot(CS_ROOT, [CS_ROOT / f for f in FILES])
    frozen.update(judge_model=read_json(CS_ROOT / CONFIG_PATH)["judge"]["model_id"], mode="decoupled_sym",
                  guided_json=True, created_at=datetime.now(timezone.utc).isoformat(),
                  scope="gold-free per-pair judgement of every pair any E2/E5b arm selected")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input_snapshot.json").write_text(json.dumps(frozen, indent=2) + "\n")


def internal_work() -> tuple[list, dict, dict]:
    rows = {json.loads(l)["image_id"]: json.loads(l) for l in (CS_ROOT / INTERNAL_ROWS).read_text().splitlines()}
    pairs = sorted({(i, r) for s in rx.e2_sets(rows).values() for i, rs in s.items() for r in rs})
    names = {Path(n).stem: n for n in read_json(CS_ROOT / "data/annotations/image_rule_gold.json")["images"]}
    facts = read_json(CS_ROOT / "results/retrieval/gold_facts_generic_en.json")
    return pairs, {i: CS_ROOT / "data/images" / n for i, n in names.items()}, \
        {i: row["image_facts"] for i, row in facts.items()}


def public_work() -> tuple[list, dict, dict]:
    import eval_rcasr_public as pub  # noqa: PLC0415

    base = CS_ROOT / PUBLIC_DIR
    rows = read_json(base / "public_selection.json")
    frozen = read_json(base / "input_snapshot.json")
    facts = {json.loads(l)["image_id"]: json.loads(l)["image_facts"]
             for l in (base / "facts.jsonl").read_text().splitlines()}
    return pub.mapped_pairs(rows), {i: CS_ROOT / s["path"] for i, s in frozen["images"].items()}, facts


def judge(population: str, workers: int) -> None:
    from backends.openai_api import OpenAIBackend  # noqa: PLC0415
    from images import load_image  # noqa: PLC0415

    verify_snapshot(CS_ROOT, read_json(OUT / "input_snapshot.json"))
    config.GUIDED_JSON = True
    model = read_json(CS_ROOT / CONFIG_PATH)["judge"]["model_id"]
    pairs, paths, facts = internal_work() if population == "internal" else public_work()
    cache = OUT / f"{population}_pairs.jsonl"
    done = {(d["image_id"], d["rule_id"]) for d in map(json.loads, cache.read_text().splitlines())} \
        if cache.exists() else set()
    backend = OpenAIBackend(model)

    def work(pair):
        return pair, rx.judge_pair(backend, load_image(paths[pair[0]]), facts[pair[0]] or [], pair[1])

    todo = [p for p in pairs if p not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, future in enumerate(as_completed([pool.submit(work, p) for p in todo]), 1):
            (image_id, rule_id), item = future.result()
            with cache.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"image_id": image_id, "rule_id": rule_id, "model": model,
                                         "guided_json": True, **item}, ensure_ascii=False) + "\n")
            if n % 100 == 0 or n == len(todo):
                print(f"{population}: judged {len(done) + n}/{len(pairs)}", flush=True)


def score_internal() -> None:
    """E2 metrics for every arm from the one frozen pair cache; closed-world precision flagged."""
    rows = {json.loads(l)["image_id"]: json.loads(l) for l in (CS_ROOT / INTERNAL_ROWS).read_text().splitlines()}
    judged = [json.loads(l) for l in (OUT / "internal_pairs.jsonl").read_text().splitlines()]
    verdicts = {(d["image_id"], d["rule_id"]): d["label"] for d in judged}
    family = {r["rule_id"]: r["rule_id"].split("-")[1] for r in read_json(CS_ROOT / "data/rules/rules_en.json")}
    sets = rx.e2_sets(rows)
    missing = sorted({(i, r) for s in sets.values() for i, rs in s.items() for r in rs} - set(verdicts))
    if missing:
        raise ValueError(f"{len(missing)} selected pairs have no verdict; finish --judge internal first")
    result = {"judge_model": sorted({d["model"] for d in judged}), "guided_json": True,
              "n_pairs_judged": len(verdicts), "parse_errors": sum(bool(d["parse_error"]) for d in judged),
              "labels": {k: sum(v == k for v in verdicts.values())
                         for k in ("non_compliant", "compliant", "need_review", None)},
              "arms": {name: rx.e2e_metrics(s, verdicts, rx.gold_statuses(CS_ROOT), family)
                       for name, s in sets.items()},
              "scope": "346 site images; cross-fitted held-out selections; one frozen per-pair judge"}
    (OUT / "e2e_results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--judge", choices=("internal", "public"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--score", action="store_true", help="E2 metrics for the internal arms")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.judge:
        judge(args.judge, args.workers)
    if args.score:
        score_internal()


if __name__ == "__main__":
    main()
