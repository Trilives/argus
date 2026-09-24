"""Scoped typed observations → subject screen and separate visual/normative truth.

No scores, model outputs, or unrecorded gold pairs are coerced into observations.
Callers must explicitly bind observations to the same entity/region/inspected scope.
"""
from __future__ import annotations

import math

from typed_predicates import atoms, evaluate

SOURCES = {'visual', 'external', 'grounding_affinity'}


def _validate_item(item: dict, table: dict) -> None:
    if item['atom'] not in table or item['source'] not in SOURCES:
        raise ValueError('unknown atom or evidence source')
    for key in ('scope_id', 'evidence_id'):
        if not isinstance(item[key], str) or not item[key].strip():
            raise ValueError(f'evidence requires a nonempty {key}')
    if type(item['observed']) is not bool:
        raise ValueError('observed must be an explicit Boolean')
    value, kind = item['value'], table[item['atom']]['value_type']
    if value is None:
        return
    if kind == 'boolean' and type(value) is not bool:
        raise ValueError('Boolean evidence cannot be a numeric confidence')
    if kind == 'number' and (type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError('numeric evidence must be finite and cannot be Boolean')
    if kind == 'number' and item.get('unit') != table[item['atom']]['unit']:
        raise ValueError('numeric evidence requires the exact declared unit; no implicit conversion')


def bind_observations(compiled: dict, rule: dict, observations: list[dict], scope_id: str) -> tuple:
    if not isinstance(scope_id, str) or not scope_id.strip():
        raise ValueError('an explicit evaluation scope is required')
    table = compiled['atoms']
    grouped, trace = {}, []
    external = {table[a]['canonical'] for a in rule['external_atoms']}
    for item in observations:
        _validate_item(item, table)
        canonical = table[item['atom']]['canonical']
        reason = None
        if item['scope_id'] != scope_id:
            reason = 'different_scope'
        elif item['source'] == 'grounding_affinity':
            reason = 'affinity_is_not_truth'
        elif canonical in external and item['source'] != 'external':
            reason = 'requires_external_evidence'
        elif not item['observed'] or item['value'] is None:
            reason = 'unobserved_or_unknown'
        trace.append({'atom': item['atom'], 'canonical': canonical, 'evidence_id': item['evidence_id'],
                      'scope_id': item['scope_id'], 'source': item['source'], 'value': item['value'],
                      'observed': item['observed'], 'unit': item.get('unit'),
                      'accepted': reason is None, 'reason': reason})
        if reason is None:
            grouped.setdefault(canonical, set()).add(item['value'])
    conflicts = sorted(k for k, values in grouped.items() if len(values) > 1)
    values = {name: next(iter(grouped[spec['canonical']])) if len(grouped.get(spec['canonical'], ())) == 1 else None
              for name, spec in table.items()}
    return values, conflicts, trace


def screen(compiled: dict, rule_id: str, observations: list[dict], scope_id: str) -> dict:
    rule = compiled['rules'][rule_id]
    values, conflicts, trace = bind_observations(compiled, rule, observations, scope_id)
    subject = evaluate(rule['gate'], values)
    visual = evaluate(rule['visual'], values) if rule['visual'] else None
    machine = evaluate(rule['machine'], values)
    status = 'need_review'
    if subject is False:
        status = 'not_applicable'
    elif subject is True and visual is not None:
        status = 'visual_violation' if visual else 'no_visual_violation'
    return {'rule_id': rule_id, 'scope_id': scope_id,
            'subject_status': 'unknown' if subject is None else 'in_scope' if subject else 'out_of_scope',
            'screening_status': status, 'visual_truth': visual, 'machine_truth': machine,
            'decision_scope': rule['decision_scope'], 'conflicts': conflicts,
            'unknown_atoms': [a for a in rule['required_atoms'] if values[a] is None],
            'unknown_subject_atoms': sorted(a for a in atoms(rule['gate']) if values[a] is None),
            'evidence_trace': trace,
            'qualification': 'Subject-domain and formula evaluation only; no full applicability or legal compliance claim.'}


def checkpoint_observations(compiled: dict, checkpoints: list[dict], scope_id: str,
                            record_id: str) -> tuple[list[dict], list[str]]:
    """Import cached model assertions for diagnostics; never treat them as gold.

    No confidence threshold and no cross-row/instance borrowing. Direct-visible is
    the model's assertion, not independent verification of visibility or absence.
    """
    observations, unused = [], []
    for index, row in enumerate(checkpoints):
        name = row.get('checkpoint')
        if not isinstance(name, str) or not name:
            raise ValueError('checkpoint requires an explicit name')
        if name not in compiled['atoms']:
            unused.append(name)
            continue
        if compiled['atoms'][name]['value_type'] != 'boolean':
            raise ValueError('Boolean checkpoint status cannot supply a numeric measurement')
        value = {'satisfied': True, 'violated': False}.get(row.get('status'))
        text = row.get('visible_evidence')
        observed = row.get('evidence_type') == 'direct_visible' and isinstance(text, str) and bool(text.strip())
        observations.append({'atom': name, 'value': value, 'source': 'visual',
                             'scope_id': scope_id, 'evidence_id': f'{record_id}:{index}',
                             'observed': observed})
    return observations, sorted(set(unused))
