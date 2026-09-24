#!/usr/bin/env python3
"""RCASR (paper-v3 §3.2–3.3, E1/E2/E4/E5/E5b): thin CLI over hash-bound stages.

Two freezes. The **evidence** freeze is label-free: it pins prompts, model files and
images, then runs the two rule-agnostic atom passes against a locally served extractor.
The **analysis** freeze pins that evidence plus the scoring code, then cross-fits and
scores. The pre-specification ``docs/paper_v3/design/RCASR_PRESPEC.md`` is fixed before
either runs.

Usage::

    uv run python experiments/retrieval/eval_rcasr.py evidence --prepare --out results/<dir>
    RCASR_EXTRACTOR_BASE_URL=http://localhost:8001/v1 \\
        uv run python experiments/retrieval/eval_rcasr.py evidence --infer --out results/<dir>
    uv run --group analysis python experiments/retrieval/eval_rcasr.py analysis --prepare ...
    uv run --group analysis python experiments/retrieval/eval_rcasr.py analysis --evaluate ...
    uv run --group analysis python experiments/retrieval/eval_rcasr.py analysis --judge ...
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

CS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CS_ROOT / "src"))

from atom_posterior import infer_evidence, prepare_evidence  # noqa: E402
from research_snapshot import read_json  # noqa: E402

CONFIG_PATH = "data/rules/proposed/rcasr_v1.json"
DEFAULT_EXTRACTOR_URL = "http://localhost:8001/v1"


def internal_images() -> dict[str, Path]:
    names = read_json(CS_ROOT / "data/annotations/image_rule_gold.json")["images"]
    return {Path(n).stem: CS_ROOT / "data/images" / n for n in sorted(names)}


def evidence(args: argparse.Namespace) -> None:
    out = CS_ROOT / args.out
    if args.prepare:
        if args.population == "internal":
            images, scope = internal_images(), "500 development-exposed internal gold images"
        else:
            from rcasr_experiment import public_images  # noqa: PLC0415
            images, scope = public_images(CS_ROOT), "ConstructionSite-10k test images with a recorded violation"
        prepare_evidence(CS_ROOT, out, CONFIG_PATH, images, scope)
        print(f"froze {len(images)} images -> {out / 'input_snapshot.json'}")
    if args.infer:
        url = os.environ.get("RCASR_EXTRACTOR_BASE_URL", DEFAULT_EXTRACTOR_URL)
        infer_evidence(CS_ROOT, out, url, workers=args.workers)


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def ci_text(ci) -> str:
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "n/a"


def render(result: dict) -> str:
    lines = ["# RCASR — site-grouped cross-fitted evaluation", "", result["scope"], "",
             f"Population: {result['population']}", "", "## Pre-specified gates", "",
             "| gate | Δ recall | 95% site CI | bar | passed |", "|---|---:|---|---|---|"]
    for name, g in result["gates"].items():
        bar = "CI > 0" if g["bar"] is None else f">= +{g['bar']:.2f}, CI > 0"
        lines.append(f"| {name} | {g['recall_delta']:+.4f} | {ci_text(g['recall_delta_ci'])} | {bar} | {g['passed']} |")
    lines += ["", "## Arms (held-out, vs matched-width prior rank mixture)", "",
              "| arm | recall | width | p95 | macro family | ref recall | Δ | 95% site CI |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name, arm in result["arms"].items():
        m, r, c = arm["metrics"], arm["reference_metrics"], arm["vs_matched_prior"]
        lines.append(f"| {name} | {pct(m['recall_micro'])} | {m['width_mean']:.3f} | {m['width_p95']:.1f} | "
                     f"{pct(m['macro_family_recall'])} | {pct(r['recall_micro'])} | {c['recall_delta']:+.4f} | "
                     f"{ci_text(c['uncertainty']['recall_delta_ci'])} |")
    lines += ["", "## Calibration (CRC on the calibration fold; held-out risk)", "",
              "| arm | α | risk | gap | width mean | p95 | deferral | infeasible folds |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, c in result["calibration"].items():
        lines.append(f"| {name} | {c['alpha']} | {c['held_out_empirical_risk']:.4f} | {c['risk_gap']:+.4f} | "
                     f"{c['width_mean']:.3f} | {c['width_p95']:.1f} | {c['deferral_rate']:.3f} | {c['infeasible_folds']} |")
    lines += ["", "## Leave-one-family-out", "", "| prior | family | held-out recall | matched prior | support |",
              "|---|---|---:|---:|---:|"]
    for prior, fams in result["family_holdout"].items():
        for fam, v in fams.items():
            lines.append(f"| {prior} | {fam} | {pct(v['held_out_family_recall'])} | "
                         f"{pct(v['matched_prior_family_recall'])} | {v['positive_support']} |")
    lines += ["", "## Harm (recorded violations the matched reference kept and RCASR dropped)", ""]
    for prior, h in result["harm"].items():
        lines.append(f"- {prior}: lost {h['lost']}, gained {h['gained']}; lost by family {h['lost_by_family']}; "
                     f"lost with an opening-class gate {h['lost_with_opening_class_gate']}")
    ev = result["evidence"]
    lines += ["", "## Evidence", "", f"- argmax/text agreement {pct(ev['argmax_text_agreement'])}; "
              f"unknown posteriors {sum(ev['unknown_posteriors'].values())}",
              f"- per image: {ev['per_image_mean']}", "", "## Limits", ""]
    lines += [f"- {x}" for x in result["limits"]]
    return "\n".join(lines) + "\n"


def analysis(args: argparse.Namespace) -> None:
    import json  # noqa: PLC0415

    import rcasr_experiment as rx  # noqa: PLC0415

    out = CS_ROOT / args.out
    if args.prepare:
        rx.prepare(CS_ROOT, out, args.evidence, CONFIG_PATH)
        (out / "analysis_plan.md").write_bytes((CS_ROOT / "docs/paper_v3/design/RCASR_PRESPEC.md").read_bytes())
        print(f"froze analysis -> {out / 'input_snapshot.json'}")
    if args.evaluate:
        result, rows = rx.evaluate(CS_ROOT, out)
        rx.write_json(out / "rcasr_results.json", result)
        (out / "rcasr_results.md").write_text(render(result), encoding="utf-8")
        with (out / "rcasr_rows.jsonl").open("w", encoding="utf-8") as handle:
            for image_id, row in sorted(rows.items()):
                handle.write(json.dumps({"image_id": image_id, **row}, ensure_ascii=False) + "\n")
        print(json.dumps(result["gates"], indent=2))
    if args.judge:
        config = read_json(CS_ROOT / CONFIG_PATH)
        rx.judge(CS_ROOT, out, config["judge"]["model_id"], workers=args.workers)
    if args.e2e:
        rx.write_json(out / "e2e_results.json", rx.evaluate_e2e(CS_ROOT, out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    ev = sub.add_parser("evidence")
    ev.add_argument("--out", required=True)
    ev.add_argument("--population", choices=("internal", "public"), default="internal")
    ev.add_argument("--prepare", action="store_true")
    ev.add_argument("--infer", action="store_true")
    ev.add_argument("--workers", type=int, default=8)
    an = sub.add_parser("analysis")
    an.add_argument("--out", required=True)
    an.add_argument("--evidence", default="results/2026-09-22_rcasr_evidence")
    for flag in ("--prepare", "--evaluate", "--judge", "--e2e"):
        an.add_argument(flag, action="store_true")
    an.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "evidence":
        evidence(args)
    else:
        analysis(args)


if __name__ == "__main__":
    main()
