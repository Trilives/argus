# `<<RETRIEVAL_TARGET_GUIDANCE>>` — English replacements

English versions of the two guidance paragraphs hardcoded in `src/prompts.py`
(`_RETRIEVAL_TARGET_GUIDANCE`). Substitute these when wiring the English prompt set.

`<<RULE_VOCABULARY>>` below is not literal text: `build_fact_messages` fills it
from `rules_en.json` at build time (every `subcategory`, lowercased and sorted —
see `prompts.rule_vocabulary`). Never expand it by hand here or in a prompt file;
a copy is what drifted from the library last time.

## text_overlap

These facts feed a one-shot text-overlap retrieval with no second chance. Use the standard construction-safety term for each object, not a colloquial paraphrase or synonym — the rule library covers `<<RULE_VOCABULARY>>`. List every object in the image that could relate to a rule — do not omit anything because its importance seems uncertain.

## agent_grep

These facts will be used by an agent that can search the rule library over multiple turns on its own. List only representative scene objects actually present, ordered by importance, without repeating the same object. You need not enumerate every detail — the agent will search further for uncertain objects.
