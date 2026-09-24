"""Semantic checks for the experimental typed overlay; no new image labels."""
import copy
import json
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from typed_predicates import parse, evaluate, atoms
from typed_schema import compile_schema
from typed_evidence import screen, checkpoint_observations
from symbolic_judgement import eval_formula
from typed_audit import replay_row, summarize


class PredicateTests(unittest.TestCase):
    def test_decimal_boundary_and_precedence(self):
        tree = parse("dust > 0.30 OR (person == yes AND distance < 0.9)")
        self.assertIs(evaluate(tree, {"dust": .3, "person": False}), False)
        self.assertIs(evaluate(tree, {"dust": .30001}), True)
        self.assertIsNone(evaluate(tree, {"dust": .3, "person": True}))
        self.assertEqual(atoms(tree), {"dust", "person", "distance"})

    def test_rejects_unconsumed_or_unsupported_syntax(self):
        for text in ["x == yes;", "x == yes # ignore", "f(x)", "x > float('nan')",
                     "x == yes OR True", "x == yes $", "x == 1", "0 < x < 1"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse(text)

    def test_rejects_wrong_value_types(self):
        for expression, value in [("x == yes", 1), ("x > 1", True), ("x > 1", float('nan'))]:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                evaluate(parse(expression), {"x": value})

    def test_boolean_equivalence_to_existing_visual_evaluator(self):
        rng = random.Random(20260919)
        rules = json.loads((ROOT / "data/rules/rules_en.json").read_text())
        statuses = {True: "satisfied", False: "violated", None: "need_review"}
        for rule in rules:
            formula = rule['visual_screening_rule']
            if formula == 'not_available_for_single_image':
                continue
            tree = parse(formula)
            for _ in range(100):
                values = {a: rng.choice([True, False, None]) for a in sorted(atoms(tree))}
                self.assertIs(evaluate(tree, values), eval_formula(formula, {a: statuses[v] for a, v in values.items()}))


class TypedSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = json.loads((ROOT / "data/rules/rules_en.json").read_text())
        cls.schema = json.loads((ROOT / "data/rules/proposed/typed_evidence_v1.json").read_text())
        cls.compiled = compile_schema(cls.rules, cls.schema)

    def rule(self, prefix):
        return next(r for r in self.compiled['rules'] if r.startswith(prefix))

    def item(self, atom, value, source='visual', scope='target-1', **kwargs):
        return dict(atom=atom, value=value, source=source, scope_id=scope,
                    evidence_id='observation-1', observed=True, **kwargs)

    def test_complete_inventory_and_conservative_aliases(self):
        self.assertEqual(self.compiled['source_atom_count'], 182)
        self.assertEqual(len(self.compiled['rules']), 42)
        table = self.schema['atoms']
        self.assertEqual(table['extinguisher_present']['canonical'], table['fire_extinguisher_present']['canonical'])
        for a, b in [('guardrail_present', 'fence_present'), ('fall_risk_present', 'fall_or_trip_risk_present'),
                     ('cover_fixed', 'closure_or_cover_secure'), ('work_at_height', 'height_over_2m')]:
            self.assertNotEqual(table[a]['canonical'], table[b]['canonical'])

    def test_absent_and_unknown_subject_never_compliant(self):
        rule = self.rule('R-BHV-002')
        self.assertEqual(screen(self.compiled, rule, [], 'target-1')['screening_status'], 'need_review')
        result = screen(self.compiled, rule, [self.item('person_present', False)], 'target-1')
        self.assertEqual(result['screening_status'], 'not_applicable')

    def test_evidence_is_scoped_and_detector_affinity_is_not_truth(self):
        rule = self.rule('R-BHV-002')
        rows = [self.item('person_present', False, scope='different-person')]
        self.assertEqual(screen(self.compiled, rule, rows, 'target-1')['subject_status'], 'unknown')
        rows = [self.item('person_present', False, source='grounding_affinity')]
        self.assertEqual(screen(self.compiled, rule, rows, 'target-1')['subject_status'], 'unknown')

    def test_external_only_predicate_ignores_visual_claim(self):
        rule = self.rule('R-CIV-006')
        rows = [self.item('construction_site_context', True), self.item('tsp_15min_avg_mg_m3', 1., unit='mg/m3')]
        result = screen(self.compiled, rule, rows, 'target-1')
        self.assertIsNone(result['machine_truth'])
        rows[-1]['source'] = 'external'
        result = screen(self.compiled, rule, rows, 'target-1')
        self.assertIs(result['machine_truth'], True)
        self.assertEqual(result['screening_status'], 'need_review')

    def test_numeric_evidence_requires_matching_unit(self):
        rows = [self.item('tsp_15min_avg_mg_m3', 1., source='external', unit='g/m3')]
        with self.assertRaises(ValueError):
            screen(self.compiled, self.rule('R-CIV-006'), rows, 'target-1')

    def test_contradictory_aliases_become_unknown(self):
        rule = self.rule('R-CIV-015')
        rows = [self.item('extinguisher_present', True), self.item('fire_extinguisher_present', False)]
        result = screen(self.compiled, rule, rows, 'target-1')
        self.assertIn('fire_extinguisher_present', result['conflicts'])

    def test_unobserved_false_is_unknown(self):
        item = self.item('person_present', False)
        item['observed'] = False
        result = screen(self.compiled, self.rule('R-BHV-002'), [item], 'target-1')
        self.assertEqual(result['subject_status'], 'unknown')

    def test_schema_rejects_missing_atom_and_state_as_subject(self):
        schema = copy.deepcopy(self.schema)
        del schema['atoms']['helmet_present']
        with self.assertRaises(ValueError):
            compile_schema(self.rules, schema)
        schema = copy.deepcopy(self.schema)
        schema['gates'][self.rule('R-BHV-002')]['formula'] = 'helmet_worn_correctly == yes'
        with self.assertRaises(ValueError):
            compile_schema(self.rules, schema)

    def test_all_unknown_library_and_evidence_trace(self):
        for rule in self.compiled['rules']:
            result = screen(self.compiled, rule, [], 'target-1')
            self.assertEqual(result['screening_status'], 'need_review')
            self.assertTrue(result['unknown_atoms'])

    def test_cache_adapter_preserves_occlusion_and_unused_fields(self):
        checkpoints = [dict(checkpoint='person_present', status='violated',
                            evidence_type='inferred', visible_evidence='Cannot see a person'),
                       dict(checkpoint='unused_field', status='satisfied')]
        rows, unused = checkpoint_observations(self.compiled, checkpoints, 'target-1', 'record-1')
        self.assertEqual(unused, ['unused_field'])
        self.assertFalse(rows[0]['observed'])
        self.assertEqual(screen(self.compiled, self.rule('R-BHV-002'), rows, 'target-1')['screening_status'], 'need_review')

    def test_cache_adapter_rejects_boolean_numeric_confusion(self):
        with self.assertRaises(ValueError):
            checkpoint_observations(self.compiled, [{'checkpoint': 'short_edge_mm', 'status': 'satisfied'}], 'x', 'r')

    def test_subject_or_is_not_flattened_into_and(self):
        rows = [self.item('vertical_opening_present', False), self.item('low_sill_opening_present', True)]
        result = screen(self.compiled, self.rule('R-OPN-004'), rows, 'target-1')
        self.assertEqual(result['subject_status'], 'in_scope')
        self.assertEqual(result['screening_status'], 'need_review')

    def test_cached_gold_status_does_not_affect_predictions(self):
        rule = self.rule('R-BHV-002')
        row = dict(image_id='image', rule_id=rule, model_label='compliant', gold_status='compliant',
                   rule_evidence=[dict(rule_id=rule, checkpoint_evidence=[])])
        first = replay_row(self.compiled, row, 0, 'hash')
        row['gold_status'] = 'non_compliant'
        self.assertEqual(first, replay_row(self.compiled, row, 0, 'hash'))

    def test_cache_failures_are_retained_in_summary_denominator(self):
        row = dict(image_id='image', rule_id=self.rule('R-BHV-002'), parse_error='bad response')
        failed = replay_row(self.compiled, row, 0, 'hash')
        summary = summarize([failed])
        self.assertEqual(summary['n_rows'], 1)
        self.assertEqual(summary['n_failed'], 1)
        self.assertEqual(summary['screening_counts'], {'need_review': 1})


if __name__ == '__main__':
    unittest.main()
