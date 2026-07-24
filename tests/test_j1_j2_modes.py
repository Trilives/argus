"""Tests for the protocol J1/J2 judgement modes (clean splits of `chain`)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CS_ROOT / "src"))

import config  # noqa: E402
from prompts import (  # noqa: E402
    build_chain_messages,
    build_facts_rules_messages,
    build_image_rules_messages,
)

RULES = [{"rule_id": "R-X-001", "rule_name": "test rule"}]
FACTS = ["standing water on the ground", "two workers without safety helmets"]


def content_types(messages: list[dict]) -> list[str]:
    return [part["type"] for part in messages[0]["content"]]


def prompt_text(messages: list[dict]) -> str:
    return next(p["text"] for p in messages[0]["content"] if p["type"] == "text")


class J1FactsRulesMessagesTest(unittest.TestCase):
    def test_carries_no_image_part(self):
        messages = build_facts_rules_messages(image_facts=FACTS, candidate_rules=RULES)
        self.assertEqual(content_types(messages), ["text"])

    def test_embeds_facts_and_rules(self):
        text = prompt_text(build_facts_rules_messages(image_facts=FACTS, candidate_rules=RULES))
        for fact in FACTS:
            self.assertIn(fact, text)
        self.assertIn("R-X-001", text)
        self.assertNotIn("<<", text, "unreplaced placeholder left in prompt")

    def test_shares_chain_output_schema_fields(self):
        text = prompt_text(build_facts_rules_messages(image_facts=FACTS, candidate_rules=RULES))
        for field in ("matched_rule", "compliance_judgement", "compliance_label",
                      "visual_checklist_alignment", "rectification_suggestion"):
            self.assertIn(field, text)


class J2ImageRulesMessagesTest(unittest.TestCase):
    def test_carries_image_part(self):
        messages = build_image_rules_messages(candidate_rules=RULES)
        self.assertEqual(content_types(messages), ["image", "text"])

    def test_embeds_rules_but_no_facts_placeholder(self):
        text = prompt_text(build_image_rules_messages(candidate_rules=RULES))
        self.assertIn("R-X-001", text)
        self.assertNotIn("<<", text, "unreplaced placeholder left in prompt")
        self.assertNotIn("pre-extracted image facts", text, "J2 must not reference Stage-1 facts")

    def test_shares_chain_output_schema_fields(self):
        text = prompt_text(build_image_rules_messages(candidate_rules=RULES))
        for field in ("matched_rule", "compliance_judgement", "compliance_label"):
            self.assertIn(field, text)


class ProtocolCodeMapTest(unittest.TestCase):
    def test_j1_j2_j3_all_mapped(self):
        self.assertEqual(config.PROTOCOL_JUDGEMENT_CODES["J1"], "facts_rules")
        self.assertEqual(config.PROTOCOL_JUDGEMENT_CODES["J2"], "image_rules")
        self.assertEqual(config.PROTOCOL_JUDGEMENT_CODES["J3"], "decoupled")

    def test_chain_hybrid_mentions_both_inputs_j_splits_do_not(self):
        chain_text = prompt_text(
            build_chain_messages(image_facts=FACTS, candidate_rules=RULES)
        )
        self.assertIn("pre-extracted image facts", chain_text)
        self.assertIn("image", chain_text)


if __name__ == "__main__":
    unittest.main()
