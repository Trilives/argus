"""Strict, non-executing parser for library predicates with unknown-valued logic."""
from __future__ import annotations

import ast
import math
import operator
import re

OPS = {ast.Eq: '==', ast.Gt: '>', ast.Lt: '<', ast.GtE: '>=', ast.LtE: '<='}
COMPARE = {'==': operator.eq, '>': operator.gt, '<': operator.lt,
           '>=': operator.ge, '<=': operator.le}


def parse(expression: str) -> tuple:
    """Accept only comparisons, parentheses, AND/OR; consume every character."""
    if not expression.strip() or re.search(r'[^A-Za-z0-9_\s().<>=]', expression):
        raise ValueError('invalid predicate characters or empty expression')
    text = re.sub(r'\b(AND|OR|yes|no)\b',
                  lambda m: {'AND': 'and', 'OR': 'or', 'yes': 'True', 'no': 'False'}[m[0]], expression)
    try:
        node = ast.parse(text.strip(), mode='eval').body
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('invalid predicate syntax') from exc
    return _convert(node)


def _convert(node: ast.AST) -> tuple:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        return ('and' if isinstance(node.op, ast.And) else 'or', tuple(map(_convert, node.values)))
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.left, ast.Name):
        raise ValueError('expected a single atom comparison')
    right, op = node.comparators[0], OPS.get(type(node.ops[0]))
    if not op or not isinstance(right, ast.Constant):
        raise ValueError('unsupported predicate operator or operand')
    value = right.value
    if op == '==':
        if type(value) is not bool:
            raise ValueError('equality expects yes/no')
    elif type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('threshold expects a finite number')
    return ('atom', node.left.id, op, value)


def atoms(tree: tuple) -> set[str]:
    if tree[0] == 'atom':
        return {tree[1]}
    return set().union(*(atoms(child) for child in tree[1]))


def comparisons(tree: tuple) -> list[tuple]:
    if tree[0] == 'atom':
        return [tree]
    return [leaf for child in tree[1] for leaf in comparisons(child)]


def evaluate(tree: tuple, values: dict) -> bool | None:
    """Strong Kleene logic; unknown values remain None, never false by default."""
    if tree[0] == 'atom':
        _, name, op, expected = tree
        value = values.get(name)
        if value is None:
            return None
        if op == '==' and type(value) is not bool:
            raise ValueError(f'{name}: expected a Boolean observation')
        if op != '==' and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f'{name}: expected a finite numeric observation')
        return COMPARE[op](value, expected)
    results = [evaluate(child, values) for child in tree[1]]
    decisive = False if tree[0] == 'and' else True
    if any(value is decisive for value in results):
        return decisive
    return None if None in results else not decisive
