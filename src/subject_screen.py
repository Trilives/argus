"""Inference-time typed subject observations, and the gate verdicts they support.

One rule-agnostic pass per image answers the 39 statements that appear in a typed
subject gate. The prompt names no rule, provision or violation. ``unclear``, a
missing id, an unparseable reply and a rejected observation all resolve to unknown,
and unknown never excludes a provision — that invariant is what separates this from
converting absence of evidence into evidence of absence.
"""
from __future__ import annotations

import json

from typed_evidence import screen
from typed_predicates import atoms

DEFAULT_CONFIG_PATH = 'data/rules/proposed/subject_screen_v1.json'
CONFIG_PATH = DEFAULT_CONFIG_PATH  # kept for callers that pin the v1 extractor
ANSWERS = {'yes': True, 'no': False, 'unclear': None}


def gate_atoms(compiled: dict) -> list[str]:
    return sorted(set().union(*(atoms(r['gate']) for r in compiled['rules'].values())))


def validate_config(config: dict, compiled: dict) -> None:
    names = gate_atoms(compiled)
    if sorted(config['questions']) != names:
        raise ValueError('the question set must cover every subject-gate atom exactly once')
    for name in names:
        spec = compiled['atoms'][name]
        if spec['external_only'] or spec['value_type'] != 'boolean':
            raise ValueError(f'subject question on an external or numeric atom: {name}')
        if not str(config['questions'][name]).strip():
            raise ValueError(f'empty question: {name}')
    if config['answer_values'] != {'yes': True, 'no': False, 'unclear': None}:
        raise ValueError('answer vocabulary must stay yes/no/unclear')
    if '{statements}' not in config['instruction_template']:
        raise ValueError('instruction template must place the statements')


def build_prompt(config: dict) -> str:
    names = sorted(config['questions'])
    statements = '\n'.join(f'  {name}: {config["questions"][name]}' for name in names)
    return config['instruction_template'].format(statements=statements, n=len(names))


def build_messages(config: dict) -> list[dict]:
    return [{'role': 'user', 'content': [{'type': 'image'},
                                         {'type': 'text', 'text': build_prompt(config)}]}]


def _first_object(text: str) -> dict:
    start = text.find('{')
    if start < 0:
        raise ValueError('reply contains no JSON object')
    depth = 0
    for index in range(start, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start:index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError('reply is not a JSON object')
                return parsed
    raise ValueError('reply has an unterminated JSON object')


def parse_reply(text: str, config: dict) -> dict:
    """Answers, unrecognised ids and missing ids; anything unusable becomes unclear."""
    parsed = _first_object(text)
    answers, unexpected, malformed = {}, [], []
    for key, value in parsed.items():
        if key not in config['questions']:
            unexpected.append(key)
            continue
        word = value.strip().lower() if isinstance(value, str) else ''
        if word not in ANSWERS:
            malformed.append(key)
            continue
        answers[key] = word
    missing = sorted(set(config['questions']) - set(answers))
    return {'answers': answers, 'missing': missing, 'malformed': sorted(malformed),
            'unexpected': sorted(set(unexpected))}


def observations(answers: dict, scope_id: str, record_id: str) -> list[dict]:
    """Typed observations for the answered atoms; unclear stays unobserved."""
    out = []
    for name, word in sorted(answers.items()):
        value = ANSWERS[word]
        out.append({'atom': name, 'value': value, 'source': 'visual', 'scope_id': scope_id,
                    'evidence_id': f'{record_id}:{name}', 'observed': value is not None})
    return out


def gate_verdicts(compiled: dict, answers: dict, scope_id: str, record_id: str) -> dict:
    """``rule_id -> in_scope | out_of_scope | unknown`` from scene evidence alone."""
    items = observations(answers, scope_id, record_id)
    return {rule_id: screen(compiled, rule_id, items, scope_id)['subject_status']
            for rule_id in sorted(compiled['rules'])}


def withhold(selected: dict, verdicts: dict) -> dict:
    """Drop only provisions the screen positively judged out of scope."""
    if set(selected) != set(verdicts):
        raise ValueError('selection and screen populations differ')
    return {image_id: [rule for rule in row if verdicts[image_id][rule] != 'out_of_scope']
            for image_id, row in selected.items()}
