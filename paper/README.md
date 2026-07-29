# Paper artifacts

## `supplement.pdf` — the online supplement

The manuscript's online supplement (4 pp). It carries:

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

ASCE stopped hosting Supplemental Materials files on 5 January 2025, so the supplement is hosted
here and the manuscript links to it by the all-versions Zenodo DOI. Nothing in it is needed to
follow the argument of the paper — it records the detail needed to audit or reproduce it.
