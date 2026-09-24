"""Frozen correspondence between the four released public rules and library outputs.

The source labels are violation-only: a mapped positive supports a detection-rate
endpoint and nothing else. An absent public rule is unknown, never compliant and
never inapplicable, and a flag on an unmapped provision is unscoreable rather than
a false positive. Nothing here reads system output.
"""
from __future__ import annotations

from collections import Counter
import re

MAP_PATH = 'data/public_eval/public_map_v1.json'
LIBRARY = 'data/rules/rules_en.json'
LABELS = 'data/public_eval/public_labels_test.jsonl'
PUBLIC_RULES = ('rule_1', 'rule_2', 'rule_3', 'rule_4')


def _check_mapping(entry: dict, library: set[str], mapped: set[str]) -> None:
    if entry['satisfaction'] != 'any_of':
        raise ValueError('only any_of satisfaction is defined; all_of would manufacture misses')
    if not entry['targets'] or not set(entry['targets']) <= library:
        raise ValueError(f"unknown or empty targets for {entry['public_rule']}")
    if len(set(entry['targets'])) != len(entry['targets']):
        raise ValueError(f"duplicate target for {entry['public_rule']}")
    for field in ('source_text', 'coverage'):
        if not str(entry.get(field, '')).strip():
            raise ValueError(f"{entry['public_rule']} needs a nonempty {field}")
    if not entry['conditions']:
        raise ValueError(f"{entry['public_rule']} needs at least one written condition")
    mapped.update(entry['targets'])


def validate(mapping: dict, rules: list[dict]) -> None:
    """Structural check only; callers verify the binding hashes separately."""
    library = {r['rule_id'] for r in rules}
    if mapping['semantics']['satisfaction'].split()[0] != 'any_of':
        raise ValueError('mapping semantics must declare any_of')
    seen = {m['public_rule'] for m in mapping['mappings']}
    if seen != set(PUBLIC_RULES) or len(mapping['mappings']) != len(PUBLIC_RULES):
        raise ValueError('every released public rule must be mapped exactly once')
    mapped: set[str] = set()
    for entry in mapping['mappings']:
        _check_mapping(entry, library, mapped)
    if mapped != set(mapping['mapped_library_rules']):
        raise ValueError('mapped_library_rules disagrees with the mapping targets')
    if mapped | set(mapping['unmapped_library_rules']) != library:
        raise ValueError('mapped and unmapped provisions do not partition the library')
    for row in mapping['derivations']:
        if row['effect'] != 'unknown' or row['public_rule'] not in seen or row['field'] != 'reason':
            raise ValueError(f"unsupported derivation: {row['id']}")
        re.compile(row['match'])
        re.compile(row['unless'])


def _derivation_hit(mapping: dict, public_rule: str, reason: str) -> str | None:
    for row in mapping['derivations']:
        if row['public_rule'] != public_rule:
            continue
        if re.search(row['match'], reason) and not re.search(row['unless'], reason):
            return row['id']
    return None


def image_targets(mapping: dict, row: dict) -> list[dict]:
    """Expand one released label row into scoreable targets and recorded exclusions.

    Only recorded positives are returned. Rules the row does not list stay absent,
    because the release cannot tell a compliant image from an inapplicable one.
    """
    entries = {m['public_rule']: m for m in mapping['mappings']}
    out = []
    for name in row['violated_public_rules']:
        public_rule = name.removesuffix('_violation')
        if public_rule not in entries:
            raise ValueError(f'unmapped public rule in labels: {name}')
        reason = row['violations'][name].get('reason') or ''
        excluded_by = _derivation_hit(mapping, public_rule, reason)
        out.append({'image_id': row['image_id'], 'public_rule': public_rule,
                    'status': 'unknown' if excluded_by else 'scoreable',
                    'targets': [] if excluded_by else list(entries[public_rule]['targets']),
                    'excluded_by': excluded_by})
    return out


def coverage(mapping: dict, rows: list[dict]) -> dict:
    """Per-rule support and exclusion counts over the released labels."""
    expanded = [item for row in rows for item in image_targets(mapping, row)]
    scoreable = [i for i in expanded if i['status'] == 'scoreable']
    return {'n_images': len(rows),
            'n_images_with_any_recorded_violation': sum(bool(r['violated_public_rules']) for r in rows),
            'n_recorded_positives': len(expanded),
            'n_scoreable_positives': len(scoreable),
            'per_rule_recorded': dict(Counter(i['public_rule'] for i in expanded)),
            'per_rule_scoreable': dict(Counter(i['public_rule'] for i in scoreable)),
            'excluded_by_derivation': dict(Counter(
                i['excluded_by'] for i in expanded if i['excluded_by'])),
            'excluded_examples': sorted((i['image_id'], i['excluded_by'])
                                        for i in expanded if i['excluded_by']),
            'n_mapped_library_rules': len(mapping['mapped_library_rules']),
            'n_unmapped_library_rules': len(mapping['unmapped_library_rules']),
            'qualification': 'Violation-only source labels; detection rate over mapped positives is '
                             'the only supported endpoint. Unmapped provisions are unscoreable, not wrong.'}
