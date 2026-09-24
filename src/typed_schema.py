"""Compile a separately versioned typed overlay without rewriting source rules."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json

from typed_predicates import atoms, comparisons, parse

KINDS = {'entity', 'activity', 'context', 'spatial', 'hazard', 'visibility',
         'document', 'violation', 'state', 'measurement'}
SUBJECT_KINDS = {'entity', 'activity', 'context', 'spatial'}
SENTINEL = 'not_available_for_single_image'


def rule_signature(rules: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rules, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _check_table(schema: dict) -> None:
    table = schema['atoms']
    aliases = {a: group['canonical'] for group in schema['aliases'] for a in group['members']}
    if len(aliases) != sum(len(g['members']) for g in schema['aliases']):
        raise ValueError('duplicate alias declaration')
    for name, spec in table.items():
        if spec['kind'] not in KINDS or spec['value_type'] not in {'boolean', 'number'}:
            raise ValueError(f'invalid atom type: {name}')
        canonical = spec['canonical']
        if canonical != aliases.get(name, name) or canonical not in table:
            raise ValueError(f'undeclared alias: {name}')
        target = table[canonical]
        if any(spec[k] != target[k] for k in ('kind', 'value_type', 'external_only')):
            raise ValueError(f'incompatible alias: {name}')
        if not spec['reason'] or not isinstance(spec['external_only'], bool):
            raise ValueError(f'incomplete atom definition: {name}')
        if spec['value_type'] == 'number' and not spec.get('unit'):
            raise ValueError(f'numeric atom requires a unit: {name}')
    if not set(aliases) <= set(table):
        raise ValueError('alias names absent from vocabulary')


def _compile_rule(rule: dict, schema: dict) -> dict:
    gate = parse(schema['gates'][rule['rule_id']]['formula'])
    machine = parse(rule['machine_rule'])
    visual = None if rule['visual_screening_rule'] == SENTINEL else parse(rule['visual_screening_rule'])
    source = atoms(machine) | (atoms(visual) if visual else set())
    required = source | atoms(gate)
    if not required <= set(schema['atoms']):
        raise ValueError(f'missing typed atoms: {sorted(required - set(schema["atoms"]))}')
    for _, name, op, expected in comparisons(gate):
        if schema['atoms'][name]['kind'] not in SUBJECT_KINDS or op != '==' or expected is not True:
            raise ValueError(f'subject gate contains a state/violation/evidence test: {name}')
    for tree in (gate, machine, visual):
        for _, name, op, _ in comparisons(tree) if tree else []:
            if schema['atoms'][name]['value_type'] != ('boolean' if op == '==' else 'number'):
                raise ValueError(f'predicate/type mismatch: {name}')
    external = set(rule.get('non_visual_fields', {})) & required
    external |= {a for a in required if schema['atoms'][a]['external_only']}
    return {'gate': gate, 'machine': machine, 'visual': visual, 'source_atoms': sorted(source),
            'required_atoms': sorted(required), 'external_atoms': sorted(external),
            'decision_scope': rule['decision_scope'], 'gate_reason': schema['gates'][rule['rule_id']]['reason']}


def compile_schema(rules: list[dict], schema: dict) -> dict:
    _check_table(schema)
    ids = {r['rule_id'] for r in rules}
    if len(ids) != len(rules) or ids != set(schema['gates']):
        raise ValueError('subject gates must match every library rule exactly once')
    if rule_signature(rules) != schema['source_rule_signature']:
        raise ValueError('overlay source library changed')
    compiled = {r['rule_id']: _compile_rule(r, schema) for r in rules}
    source = set().union(*(set(r['source_atoms']) for r in compiled.values()))
    required = set().union(*(set(r['required_atoms']) for r in compiled.values()))
    if required != set(schema['atoms']):
        raise ValueError('unused or missing vocabulary entries')
    for atom in source:
        users = {k for k, row in compiled.items() if atom in row['source_atoms']}
        if users != set(schema['atoms'][atom]['source_rules']):
            raise ValueError(f'atom source-rule provenance mismatch: {atom}')
    return {'version': schema['version'], 'atoms': schema['atoms'], 'rules': compiled,
            'source_atom_count': len(source), 'new_subject_atoms': sorted(required - source)}


def inventory(compiled: dict) -> dict:
    reuse = defaultdict(set)
    for rule_id, row in compiled['rules'].items():
        for name in row['required_atoms']:
            reuse[compiled['atoms'][name]['canonical']].add(rule_id)
    source = set().union(*(set(r['source_atoms']) for r in compiled['rules'].values()))
    return {'n_rules': len(compiled['rules']), 'source_atom_count': len(source),
            'source_value_types': dict(Counter(compiled['atoms'][a]['value_type'] for a in sorted(source))),
            'typed_atom_count': len(compiled['atoms']), 'canonical_count': len(reuse),
            'new_subject_atoms': compiled['new_subject_atoms'],
            'kinds': dict(Counter(row['kind'] for row in compiled['atoms'].values())),
            'canonical_rule_reuse': {k: sorted(v) for k, v in sorted(reuse.items())},
            'shared_canonicals': sum(len(v) > 1 for v in reuse.values()),
            'rules': compiled['rules']}
