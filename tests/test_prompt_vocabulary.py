"""Prompts may only name vocabulary and rule_ids the rule library contains.

The fact cache in `results/retrieval/gold_facts_generic_en.json` describes
objects the corpus has no word for ("distribution box", 20x) because the Stage-1
prompt taught them. Every fact spent on such a term is unreachable by retrieval,
for every model and every prompt, and nothing failed loudly. These tests close
that loop: each term a Stage-1 prompt names goes through the same grep the agent
calls, so a prompt cannot recommend a word no rule contains. The same drift
reaches rule_ids: prompts illustrate their output format with ids, and those go
stale silently when the library is renumbered.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

CS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CS_ROOT / "src"))

from prompts import build_fact_messages, rule_vocabulary  # noqa: E402
from retrieval.agent_grep import RuleCorpus  # noqa: E402

RULES_PATH = CS_ROOT / "data/rules/rules_en.json"

# Where a Stage-1 prompt enumerates example objects. Extraction is anchored to
# these phrasings rather than to "any parenthesised list", because the prompts
# also parenthesise verdict words and non-visual conditions -- terms that must
# NOT appear in the corpus.
_OBJECT_LIST_PATTERNS = (
    (r"names a specific object \(([^)]*)\)", ","),
    (r"visible objects with parts/materials, e\.g\. ([^\"]*)", ";"),
)

FACT_MODES = ("generic", "scene")
RETRIEVAL_METHODS = ("text_overlap", "agent_grep")


def prompt_text(messages: list[dict]) -> str:
    return next(part["text"] for part in messages[0]["content"] if part["type"] == "text")


def example_terms(prompt: str) -> list[str]:
    terms = []
    for pattern, separator in _OBJECT_LIST_PATTERNS:
        for listing in re.findall(pattern, prompt):
            for item in listing.split(separator):
                # "cover plate (wooden)" -> "cover plate": the material aside is
                # illustrative, the object is what retrieval has to match.
                term = re.sub(r"\([^)]*\)", "", item).strip().rstrip(".")
                if term and term not in {"etc", "..."}:
                    terms.append(term)
    return terms


class Stage1PromptVocabularyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = RuleCorpus(rules=json.loads(RULES_PATH.read_text(encoding="utf-8")))

    def assert_greppable(self, term: str, *, source: str) -> None:
        self.assertGreater(
            len(self.corpus.grep(term)), 0,
            f"{source} teaches '{term}', which greps to zero rules",
        )

    def test_derived_rule_vocabulary_is_greppable(self) -> None:
        """The guidance's term list is derived from the library, so it can only
        name reachable terms -- unless the derivation reads a field grep does not
        search. Pin that: this fails the day `subcategory` leaves the grep block."""
        vocabulary = rule_vocabulary()
        self.assertGreater(len(vocabulary), 0)
        for term in vocabulary:
            with self.subTest(term=term):
                self.assert_greppable(term, source="the derived rule vocabulary")

    def test_every_example_object_a_stage1_prompt_teaches_is_greppable(self) -> None:
        for mode in FACT_MODES:
            for method in RETRIEVAL_METHODS:
                prompt = prompt_text(build_fact_messages(mode=mode, retrieval_method=method))
                terms = example_terms(prompt)
                with self.subTest(mode=mode, retrieval_method=method):
                    # Without this the whole test passes silently once a prompt
                    # is reworded past the extraction patterns above.
                    self.assertGreater(len(terms), 0, "no example objects extracted")
                for term in terms:
                    with self.subTest(mode=mode, retrieval_method=method, term=term):
                        self.assert_greppable(term, source=f"the {mode} Stage-1 prompt")

    def test_no_stage1_prompt_leaves_a_placeholder_unfilled(self) -> None:
        for mode in FACT_MODES:
            for method in RETRIEVAL_METHODS:
                prompt = prompt_text(build_fact_messages(mode=mode, retrieval_method=method))
                with self.subTest(mode=mode, retrieval_method=method):
                    self.assertNotIn("<<", prompt)


class PromptRuleIdTest(unittest.TestCase):
    """Every rule_id any English prompt shows must exist in the live library.

    `tests/test_agent_prompt_ab_task.py` pins this for the A/B driver prompts and
    the agent prompt reference; this covers the prompt files the pipeline itself
    loads, where `R-opening-001` survived a renumbering to `R-OPN-001-*`.
    """

    def test_rule_ids_shown_in_prompt_files_exist(self) -> None:
        known_ids = {rule["rule_id"] for rule in json.loads(RULES_PATH.read_text(encoding="utf-8"))}
        found = False
        for path in sorted((CS_ROOT / "Prompts_en").rglob("*.md")):
            shown = {
                match.group(0).rstrip(".,;:)")
                for match in re.finditer(r"R-[A-Za-z]+-\d{3}[a-zA-Z0-9-]*", path.read_text(encoding="utf-8"))
            }
            found = found or bool(shown)
            with self.subTest(prompt=path.name):
                self.assertEqual(set(), shown - known_ids)
        self.assertTrue(found, "no rule_ids extracted from any prompt file")


if __name__ == "__main__":
    unittest.main()
