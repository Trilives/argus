# Paper artifacts

The version of record is the [Buildings article](https://doi.org/10.3390/buildings16224502),
published on 11 November 2026. Its final supplement is hosted by
[the journal](https://www.mdpi.com/article/10.3390/buildings16224502/s1).
The local PDF described below is a retained supplement from the earlier ASCE submission stage of this
study. It is historical supporting material; use the journal supplement for the published article.

## `supplement.pdf` — earlier-stage supplement

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

This file was hosted here for the earlier ASCE submission and linked by the all-versions Zenodo DOI.
It is retained to preserve that stage's audit and reproduction detail.
