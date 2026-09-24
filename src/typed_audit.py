"""Frozen ontology audit and gold-conditioned cached evidence diagnostics."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from research_snapshot import read_json, sha256, snapshot, verify_snapshot
from typed_evidence import checkpoint_observations, screen
from typed_schema import compile_schema, inventory

CACHE = 'results/judgement/oracle_pairs_decoupled.jsonl'
LIBRARY = 'data/rules/rules_en.json'
PLAN = 'docs/Records/2026-09-19/typed_evidence_plan.md'


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def prepare(root: Path, schema_path: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('use a new empty output directory; freezes cannot be replaced')
    schema = read_json(schema_path)
    if sha256(root / LIBRARY) != schema['source_library_sha256']:
        raise ValueError('source library bytes differ from overlay')
    compile_schema(read_json(root / LIBRARY), schema)
    files = [root / name for name in (LIBRARY, CACHE, PLAN, 'pyproject.toml', 'uv.lock',
             'experiments/pipeline/check_atom_vocabulary.py', 'src/research_snapshot.py',
             'tests/test_typed_evidence.py')]
    files += [schema_path, *sorted((root / 'src').glob('typed_*.py'))]
    frozen = snapshot(root, files)
    frozen.update(schema_path=str(schema_path.relative_to(root)),
                  scope='structural checks and gold-conditioned cached J3 diagnostic; no new inference')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'input_snapshot.json', frozen)
    (out / 'analysis_plan.md').write_bytes((root / PLAN).read_bytes())
    (out / 'typed_evidence_schema.json').write_bytes(schema_path.read_bytes())


def replay_row(compiled: dict, row: dict, index: int, cache_hash: str) -> dict:
    image, rule = row['image_id'], row['rule_id']
    base = {'image_id': image, 'rule_id': rule, 'cached_model_label': row.get('model_label')}
    try:
        if row.get('run_error') or row.get('parse_error'):
            raise ValueError('source cache records an inference/parse failure')
        entries = [r for r in row.get('rule_evidence', []) if r.get('rule_id') == rule]
        if len(entries) != 1:
            raise ValueError('exactly one matching cached rule-evidence record required')
        scope = f'{image}:{rule}:cached_aggregate'
        obs, unused = checkpoint_observations(compiled, entries[0]['checkpoint_evidence'], scope,
                                             f'{cache_hash}:row-{index}')
        result = screen(compiled, rule, obs, scope)
        return {**base, 'status': 'ok', 'unused_checkpoints': unused, **result}
    except (KeyError, TypeError, ValueError) as exc:
        return {**base, 'status': 'failed', 'screening_status': 'need_review',
                'failure': type(exc).__name__ + ': ' + str(exc)}


def summarize(rows: list[dict]) -> dict:
    by_rule, rejected = defaultdict(Counter), Counter()
    for row in rows:
        by_rule[row['rule_id']][row['screening_status']] += 1
        for item in row.get('evidence_trace', []):
            if not item['accepted']:
                rejected[item['reason']] += 1
    return {'n_rows': len(rows), 'n_images': len({r['image_id'] for r in rows}),
            'n_failed': sum(r['status'] != 'ok' for r in rows),
            'screening_counts': dict(Counter(r['screening_status'] for r in rows)),
            'subject_counts': dict(Counter(r.get('subject_status', 'failed') for r in rows)),
            'cached_model_to_screen': dict(Counter(f"{r['cached_model_label']} -> {r['screening_status']}" for r in rows)),
            'formula_truth_to_screen': dict(Counter(f"{r.get('visual_truth')} -> {r['screening_status']}" for r in rows)),
            'rejected_observations': dict(rejected),
            'unused_checkpoints': dict(Counter(a for r in rows for a in r.get('unused_checkpoints', []))),
            'by_rule': {k: dict(v) for k, v in sorted(by_rule.items())},
            'qualification': 'Gold-conditioned aggregate model assertions; no entity-binding, absence-completeness, or predictive-gain claim.'}


def run(root: Path, out: Path) -> None:
    frozen = read_json(out / 'input_snapshot.json')
    verify_snapshot(root, frozen)
    compiled = compile_schema(read_json(root / LIBRARY), read_json(root / frozen['schema_path']))
    source = [json.loads(line) for line in (root / CACHE).read_text().splitlines() if line.strip()]
    keys = [(r['image_id'], r['rule_id']) for r in source]
    if len(keys) != len(set(keys)) or any(r not in compiled['rules'] for _, r in keys):
        raise ValueError('duplicate cached pair or unsupported rule ID')
    cache_hash = sha256(root / CACHE)
    rows = [replay_row(compiled, row, i, cache_hash) for i, row in enumerate(source)]
    empty = [screen(compiled, r, [], 'synthetic-empty')['screening_status'] for r in compiled['rules']]
    report = {'scope': frozen['scope'], 'snapshot_sha256': sha256(out / 'input_snapshot.json'),
              'inventory': inventory(compiled), 'all_unknown_screening_counts': dict(Counter(empty)),
              'cached_diagnostic': summarize(rows)}
    write_json(out / 'typed_audit.json', report)
    with (out / 'cached_screening.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    (out / 'typed_audit.md').write_text(render(report), encoding='utf-8')
    print(render(report))


def render(report: dict) -> str:
    inv, cache = report['inventory'], report['cached_diagnostic']
    lines = ['# Typed evidence overlay audit', '', report['scope'], '',
             f"Rules: {inv['n_rules']}; source atoms: {inv['source_atom_count']}; typed names: {inv['typed_atom_count']}.",
             f"Canonicals: {inv['canonical_count']}; shared across rules: {inv['shared_canonicals']}.",
             f"All-unknown outcomes: {report['all_unknown_screening_counts']}.", '',
             '## Cached conditional diagnostic', '', cache['qualification'], '',
             f"Rows: {cache['n_rows']}; images: {cache['n_images']}; failures retained: {cache['n_failed']}.", '',
             '| screening outcome | count |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k, v in sorted(cache['screening_counts'].items())]
    lines += ['', 'No new labels/model calls; cached model assertions are not factual gold.',
              '`no_visual_violation` does not mean normative compliance.',
              'See JSON for per-rule reuse, external fields, rejected evidence and transitions.', '']
    return '\n'.join(lines)
