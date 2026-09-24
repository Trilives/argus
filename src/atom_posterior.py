"""Continuous atom posteriors from rule-agnostic VLM passes, read off token log-probs.

Two passes per image: the subject questions frozen in ``subject_screen_v2.json`` and the
state questions frozen in ``rcasr_v1.json``. Neither prompt names a rule, provision or
violation. At each answer token the top log-probs are bucketed into yes/no/unclear by
prefix; an id with no answer token, or no mass on any of the three, stays unknown.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from threading import Lock
import time

from PIL import Image

from images import encode_image_data_url
from research_snapshot import read_json, sha256, snapshot, verify_snapshot
from subject_screen import parse_reply
from typed_predicates import atoms
from typed_schema import compile_schema

WORDS = ("yes", "no", "unclear")
BANNED = ("rule", "provision", "violation", "compliant", "regulation", "r-civ", "r-bhv",
          "r-edg", "r-opn")
_KEY_BEFORE_VALUE = re.compile(r'"([A-Za-z0-9_]+)"\s*:\s*"?\s*$')


def state_atoms(compiled: dict) -> list[str]:
    """Atoms in any visual screening formula that no subject gate already asks about."""
    gate = set().union(*(atoms(r["gate"]) for r in compiled["rules"].values()))
    visual = set().union(*(atoms(r["visual"]) for r in compiled["rules"].values() if r["visual"]))
    return sorted(visual - gate)


def validate_state(config: dict, compiled: dict) -> None:
    state = config["passes"]["state"]
    asked, skipped = set(state["questions"]), set(state["unknown_only"])
    if asked & skipped or sorted(asked | skipped) != state_atoms(compiled):
        raise ValueError("state questions must cover every non-gate visual atom exactly once")
    for name in asked:
        spec = compiled["atoms"][name]
        if spec["external_only"] or spec["value_type"] != "boolean":
            raise ValueError(f"state question on an external or numeric atom: {name}")
        text = state["questions"][name].lower()
        if not text.strip() or any(word in text for word in BANNED):
            raise ValueError(f"empty question or one that names the target: {name}")
    for name in skipped:
        if not compiled["atoms"][name]["external_only"]:
            raise ValueError(f"only external atoms may be left unasked: {name}")


def build_prompt(questions: dict, template: str) -> str:
    names = sorted(questions)
    statements = "\n".join(f"  {name}: {questions[name]}" for name in names)
    return template.format(statements=statements, n=len(names))


def request(client, model: str, prompt: str, data_url: str, decoding: dict, max_tokens: int):
    """One greedy chat completion with top-k log-probs; the raw response is returned."""
    return client.chat.completions.create(
        model=model, temperature=decoding["temperature"], max_tokens=max_tokens,
        logprobs=True, top_logprobs=decoding["top_logprobs"],
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}},
                                               {"type": "text", "text": prompt}]}],
        extra_body={"repetition_penalty": decoding["repetition_penalty"],
                    "chat_template_kwargs": {"enable_thinking": decoding["enable_thinking"]},
                    "mm_processor_kwargs": {"min_pixels": decoding["min_pixels"],
                                            "max_pixels": decoding["max_pixels"]}})


def bucket(alternatives: list[tuple[str, float]]) -> dict | None:
    """Renormalised yes/no/unclear mass over one position's top tokens."""
    mass = dict.fromkeys(WORDS, 0.0)
    for token, logprob in alternatives:
        word = token.strip().strip('"').lower()
        for target in WORDS:
            if word and target.startswith(word):
                mass[target] += math.exp(logprob)
                break
    total = sum(mass.values())
    if total <= 0:
        return None
    return {f"p_{w}": mass[w] / total for w in WORDS}


def answer_positions(tokens: list[tuple[str, list[tuple[str, float]]]], ids: set[str]) -> dict:
    """First answer token per id: the first non-quote token after ``"id": "``."""
    text, found = "", {}
    for token, alternatives in tokens:
        stripped = token.strip().lstrip('"')
        match = _KEY_BEFORE_VALUE.search(text) if stripped else None
        if match and match[1] in ids and match[1] not in found:
            found[match[1]] = alternatives
        text += token
    return found


def posteriors(tokens: list[tuple[str, list[tuple[str, float]]]], questions: dict) -> dict:
    """``atom -> {p_yes, p_no, p_unclear}`` or ``None`` (unknown) for every asked id."""
    found = answer_positions(tokens, set(questions))
    return {name: bucket(found[name]) if name in found else None for name in sorted(questions)}


def tokens_of(response) -> list[tuple[str, list[tuple[str, float]]]]:
    content = response.choices[0].logprobs.content if response.choices[0].logprobs else []
    return [(t.token, [(a.token, a.logprob) for a in t.top_logprobs]) for t in content or []]


def read_pass(response, questions: dict) -> dict:
    """Posteriors plus the argmax parse, kept side by side for a consistency audit."""
    text = response.choices[0].message.content or ""
    post = posteriors(tokens_of(response), questions)
    try:
        parsed = parse_reply(text, {"questions": questions})
        status = "ok"
    except ValueError as exc:
        parsed, status = {"answers": {}, "missing": sorted(questions)}, f"parse_error: {exc}"
    usage = response.usage
    return {"raw": text, "status": status, "answers": parsed["answers"],
            "missing": parsed["missing"], "posteriors": post,
            "finish_reason": response.choices[0].finish_reason,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None)}


# --- evidence freeze: label-free, hash-bound, resumable --------------------------------

EVIDENCE_FILES = ("src/atom_posterior.py", "src/subject_screen.py", "src/typed_schema.py",
                  "src/typed_predicates.py", "src/typed_evidence.py", "src/research_snapshot.py",
                  "src/images.py", "data/rules/rules_en.json",
                  "data/rules/proposed/typed_evidence_v1.json", "pyproject.toml", "uv.lock")


def pass_specs(root: Path, config: dict) -> dict:
    """``name -> (questions, template, max_tokens)`` for the two frozen passes."""
    sub_path = root / config["passes"]["subject"]["questions_from"]
    if sha256(sub_path) != config["passes"]["subject"]["questions_sha256"]:
        raise ValueError("subject question file changed since rcasr_v1 was frozen")
    sub, state = read_json(sub_path), config["passes"]["state"]
    tokens = config["decoding"]["max_tokens"]
    return {"subject": (sub["questions"], sub["instruction_template"], tokens["subject"]),
            "state": (state["questions"], state["instruction_template"], tokens["state"])}


def prepare_evidence(root: Path, out: Path, config_path: str, images: dict[str, Path],
                     scope: str) -> None:
    """Hash every input the evidence depends on; refuses to overwrite a freeze."""
    if (out / "input_snapshot.json").exists():
        raise FileExistsError("existing freeze cannot be overwritten; use a new output directory")
    config = read_json(root / config_path)
    compiled = compile_schema(read_json(root / "data/rules/rules_en.json"),
                              read_json(root / config["source_overlay"]))
    validate_state(config, compiled)
    pass_specs(root, config)
    names = [config_path, config["passes"]["subject"]["questions_from"], config["plan_path"],
             *config["plan_addenda"], *EVIDENCE_FILES]
    model_dir = root / config["extractor"]["model_dir"]
    frozen = snapshot(root, [root / n for n in names] + [p for p in model_dir.iterdir() if p.is_file()])
    frozen.update(config_path=config_path, scope=scope, model_id=config["extractor"]["model_id"],
                  images={i: {"path": str(p.relative_to(root)), "sha256": sha256(p)}
                          for i, p in sorted(images.items())},
                  created_at=datetime.now(timezone.utc).isoformat())
    out.mkdir(parents=True, exist_ok=True)
    (out / "input_snapshot.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")


def read_evidence(path: Path, frozen: dict, stamp: str, require_complete: bool = True) -> dict:
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            image_id = row["image_id"]
            spec = frozen["images"].get(image_id)
            if image_id in records or spec is None:
                raise ValueError(f"duplicate or unexpected evidence row: {image_id}")
            if row["snapshot_sha256"] != stamp or row["image_sha256"] != spec["sha256"]:
                raise ValueError(f"evidence row not stamped by this freeze: {image_id}")
            records[image_id] = row
    if require_complete and set(records) != set(frozen["images"]):
        raise ValueError("evidence is incomplete for the frozen population")
    return records


def _one_image(root: Path, client, spec: dict, passes: dict, config: dict) -> dict:
    path = root / spec["path"]
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"image changed after freeze: {path}")
    with Image.open(path) as handle:
        data_url = encode_image_data_url(handle.convert("RGB"), fmt="PNG")
    out = {}
    for name, (questions, template, max_tokens) in passes.items():
        started = time.perf_counter()
        response = request(client, config["extractor"]["served_model_name"],
                           build_prompt(questions, template), data_url, config["decoding"], max_tokens)
        out[name] = {**read_pass(response, questions), "seconds": time.perf_counter() - started}
    return out


def infer_evidence(root: Path, out: Path, base_url: str, workers: int = 8) -> None:
    """Both passes for every frozen image not yet in the cache; append-only, resumable."""
    from openai import OpenAI

    frozen = read_json(out / "input_snapshot.json")
    verify_snapshot(root, frozen)
    config = read_json(root / frozen["config_path"])
    stamp, cache = sha256(out / "input_snapshot.json"), out / "atom_evidence.jsonl"
    done = read_evidence(cache, frozen, stamp, require_complete=False)
    todo = [i for i in sorted(frozen["images"]) if i not in done]
    passes, client, lock = pass_specs(root, config), OpenAI(base_url=base_url, api_key="EMPTY"), Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one_image, root, client, frozen["images"][i], passes, config): i
                   for i in todo}
        for count, future in enumerate(as_completed(futures), 1):
            image_id = futures[future]
            row = {"image_id": image_id, "snapshot_sha256": stamp,
                   "image_sha256": frozen["images"][image_id]["sha256"], **future.result()}
            with lock, cache.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            if count % 25 == 0 or count == len(todo):
                print(f"evidence {len(done) + count}/{len(frozen['images'])}", flush=True)
    verify_snapshot(root, frozen)
