"""The frozen public rule mapping must stay conservative and hash-bound.

These tests read the released labels but never a system prediction, and they assert
the properties that keep §E5b honest: the mapping partitions the library, a broad
source rule is never expanded into positives for each narrower provision, an absent
public rule stays unknown, and the declared bytes are the bytes on disk.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from public_map import (LABELS, LIBRARY, MAP_PATH, PUBLIC_RULES, coverage,  # noqa: E402
                        image_targets, validate)
from research_snapshot import sha256  # noqa: E402
from typed_schema import rule_signature  # noqa: E402


# The ConstructionSite-10k labels are not redistributed (CC BY-NC, gated access); tests
# that read them run only where the file has been obtained from its authors.
HAS_LABELS = (ROOT / LABELS).exists()
NEEDS_LABELS = "needs data/public_eval/public_labels_test.jsonl (not redistributed)"


def load_labels() -> list[dict]:
    text = (ROOT / LABELS).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class MappingIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapping = json.loads((ROOT / MAP_PATH).read_text(encoding="utf-8"))
        cls.rules = json.loads((ROOT / LIBRARY).read_text(encoding="utf-8"))

    def test_mapping_validates_against_the_current_library(self) -> None:
        validate(self.mapping, self.rules)

    @unittest.skipUnless(HAS_LABELS, NEEDS_LABELS)
    def test_binding_hashes_match_the_files_on_disk(self) -> None:
        binding = self.mapping["binding"]
        self.assertEqual(sha256(ROOT / LIBRARY), binding["library_sha256"])
        self.assertEqual(sha256(ROOT / LABELS), binding["public_labels_sha256"])
        self.assertEqual(rule_signature(self.rules), binding["source_rule_signature"])

    def test_mapped_and_unmapped_partition_the_library(self) -> None:
        mapped = set(self.mapping["mapped_library_rules"])
        unmapped = set(self.mapping["unmapped_library_rules"])
        self.assertEqual(mapped & unmapped, set())
        self.assertEqual(mapped | unmapped, {r["rule_id"] for r in self.rules})

    def test_every_mapping_is_any_of_with_written_conditions(self) -> None:
        for entry in self.mapping["mappings"]:
            with self.subTest(entry["public_rule"]):
                self.assertEqual(entry["satisfaction"], "any_of")
                self.assertTrue(entry["conditions"])
                self.assertTrue(entry["source_text"].strip())

    def test_harness_rule_does_not_claim_any_edge_provision(self) -> None:
        entry = next(m for m in self.mapping["mappings"] if m["public_rule"] == "rule_2")
        self.assertEqual(entry["targets"], ["R-BHV-001-no-harness-at-height"])
        self.assertTrue(any("precondition" in text for text in entry["exclusions"]))

    def test_all_of_satisfaction_is_rejected(self) -> None:
        broken = copy.deepcopy(self.mapping)
        broken["mappings"][0]["satisfaction"] = "all_of"
        with self.assertRaises(ValueError):
            validate(broken, self.rules)

    def test_target_outside_the_library_is_rejected(self) -> None:
        broken = copy.deepcopy(self.mapping)
        broken["mappings"][0]["targets"] = ["R-NOT-A-RULE"]
        with self.assertRaises(ValueError):
            validate(broken, self.rules)

    def test_dropping_a_public_rule_is_rejected(self) -> None:
        broken = copy.deepcopy(self.mapping)
        broken["mappings"] = broken["mappings"][:3]
        with self.assertRaises(ValueError):
            validate(broken, self.rules)


@unittest.skipUnless(HAS_LABELS, NEEDS_LABELS)
class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapping = json.loads((ROOT / MAP_PATH).read_text(encoding="utf-8"))
        cls.rows = load_labels()

    def test_absent_public_rule_produces_no_target(self) -> None:
        clean = next(r for r in self.rows if not r["violated_public_rules"])
        self.assertEqual(image_targets(self.mapping, clean), [])

    def test_recorded_positive_expands_to_its_declared_targets(self) -> None:
        row = next(r for r in self.rows if r["violated_public_rules"] == ["rule_3_violation"])
        items = image_targets(self.mapping, row)
        self.assertEqual(len(items), 1)
        if items[0]["status"] == "scoreable":
            entry = next(m for m in self.mapping["mappings"] if m["public_rule"] == "rule_3")
            self.assertEqual(items[0]["targets"], entry["targets"])

    def test_eye_protection_only_positive_is_unknown_not_a_miss(self) -> None:
        row = {"image_id": "synthetic", "violated_public_rules": ["rule_1_violation"],
               "violations": {"rule_1_violation": {
                   "reason": "The worker is not wearing safety glasses when grinding."}}}
        item = image_targets(self.mapping, row)[0]
        self.assertEqual(item["status"], "unknown")
        self.assertEqual(item["excluded_by"], "rule_1_eye_face_only")
        self.assertEqual(item["targets"], [])

    def test_eye_protection_with_a_helmet_failure_stays_scoreable(self) -> None:
        row = {"image_id": "synthetic", "violated_public_rules": ["rule_1_violation"],
               "violations": {"rule_1_violation": {
                   "reason": "The person is not wearing a hard hat nor having face or eye "
                             "protections when drilling."}}}
        item = image_targets(self.mapping, row)[0]
        self.assertEqual(item["status"], "scoreable")
        self.assertEqual(item["targets"], ["R-BHV-002-helmet-and-attire"])

    def test_natural_ground_edge_is_unknown_but_excavation_slope_is_not(self) -> None:
        cases = {"The edge of the cliff is not fenced.": "unknown",
                 "No edge protection nor edge warning is used for the slope.": "unknown",
                 "No visible edge protection on the right embankment of the excavation.": "scoreable"}
        for reason, expected in cases.items():
            with self.subTest(reason):
                row = {"image_id": "synthetic", "violated_public_rules": ["rule_3_violation"],
                       "violations": {"rule_3_violation": {"reason": reason}}}
                self.assertEqual(image_targets(self.mapping, row)[0]["status"], expected)

    def test_coverage_matches_the_released_support(self) -> None:
        report = coverage(self.mapping, self.rows)
        self.assertEqual(report["n_images"], 3004)
        self.assertEqual(report["n_images_with_any_recorded_violation"], 412)
        self.assertEqual(report["per_rule_recorded"],
                         {"rule_1": 324, "rule_2": 25, "rule_3": 63, "rule_4": 24})
        self.assertLessEqual(report["n_scoreable_positives"], report["n_recorded_positives"])
        for name in PUBLIC_RULES:
            with self.subTest(name):
                self.assertLessEqual(report["per_rule_scoreable"].get(name, 0),
                                     report["per_rule_recorded"][name])


if __name__ == "__main__":
    unittest.main()
