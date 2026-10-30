# Combined original-image manifest

`manifest.csv` describes 1,000 original-image IDs: 500 `gold` and 500 `research_unlabelled`. There are 968 distinct source byte streams and 32 source-byte alias pairs. This is a fixed selection, not a uniform random sample or newly annotated gold set. No crops are included.

`source_sha256` identifies the original private input; `sha256` identifies the released JPEG. `source_byte_alias_of` refers to the canonical anonymous ID for identical source bytes, regardless of any differences in human-selected anonymisation. Original paths and site identities are not included.

`license` and `license_url` identify the terms for each file. `prior_public_release` records
only the grant category, without disclosing private source-to-demo ID mappings:

| Original-image files | Licence | Scope |
|---|---|---|
| 500 gold + 19 additional files with published gold sources | CC-BY-4.0 | Existing grant preserved |
| 18 additional files with published demonstration sources | Apache-2.0 | Existing grant preserved |
| 463 eligible additional files | CC-BY-NC-4.0 | Non-commercial use; commercial use needs separate permission |

The photographs are evaluation/supporting material, not a claimed dataset contribution.

The existing gold files and annotations are unchanged. See [gold images](../gold_images/README.md) and [the research-image data card](../research_images/README.md) for privacy processing and licence details.
