# Paper artifacts

## `supplement.pdf` — the online supplement

The manuscript's online supplement (8 pp). It carries:

1. **Rule-unit anatomy** — the typed fields of one provision unit (`R-BHV-002`, the helmet
   provision) and the five-link auditable chain they expand into.
2. **Retrieval loop** — pseudocode for the bounded `grep` / `read` / `submit` retriever (R3/R4)
   with its anytime-termination behaviour. The reference implementation is
   `src/retrieval/agent_grep.py` and `src/retrieval/agent_grep_visual.py`.
3. **Per-provision provenance** — one row per provision for all 42, giving the governing standard
   and provenance tag alongside the screening-predicate derivation (`clause` vs. `proxy`, dropped
   normative fields, `decision_scope`). The derived columns are read straight off
   `data/rules/rules_en.json`, so they cannot drift from the library the pipeline runs; the same
   table is machine-readable at `results/data_audit/rule_provenance.json`.
4. **Dedicated-detector baseline** — the head-to-head against a supervised hard-hat detector on the
   one provision a binary class scheme can address, with example violations such a scheme cannot
   express. Machine-readable at `results/external/detector_baseline_summary.json`.
5. **Unabridged worked-case chains** — the three photographs of the manuscript's case figure carried
   through the full chain with nothing abridged: every retrieved candidate, the complete clause text
   and screening formula, every checkpoint evidence line with its status, evidence type and
   confidence, the verdict, the gold status and the rectification advice. All values are read from
   the cached headline run, so they are what the pipeline produced rather than a paraphrase.

ASCE stopped hosting Supplemental Materials files on 5 January 2025, so the supplement is hosted
here and the manuscript links to it by the all-versions Zenodo DOI. Nothing in it is needed to
follow the argument of the paper — it records the detail needed to audit or reproduce it.
