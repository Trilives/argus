# Additional research photographs

**Non-commercial use only for the 463 CC-BY-NC-4.0 files.** Commercial use of those files
requires separate written permission from the company. [LICENSE](LICENSE) defines the scope;
[LICENSE_EXCEPTIONS.csv](LICENSE_EXCEPTIONS.csv) lists 19 files retaining CC BY 4.0 and 18
retaining Apache-2.0. Their earlier permissions are preserved. The `license` field in
[the combined manifest](../image_pool/manifest.csv) is authoritative for each image.

This subset contains 500 additional original photographs. Together with the unchanged 500 gold photographs, the pool has 1,000 original-image IDs representing 968 distinct source byte streams. No crops are included. The fixed handoff selection is not a uniform random sample or newly annotated evaluation gold. The new photographs have no released gold labels.

Every addition was individually reviewed by a person who selected any needed blur regions; an empty selection records a reviewed image needing no blur. Regions may be rotated. Gaussian blur is restricted to the selected pixels, followed by scaling to at most 2048 pixels on the longest side and JPEG encoding without EXIF/GPS, comments or ICC profiles. `anonymisation.json` records processing checks; person detections assist review and do not constitute automated privacy clearance.

The [combined manifest](../image_pool/manifest.csv) records raw-input `source_sha256` separately from anonymised-file `sha256`, dimensions, subset and source-byte aliases. The frozen gold photographs, annotations, splits, predictions and reported results remain unchanged. Released pixels differ from the original experimental inputs.

China Construction Third Engineering Bureau (Shenzhen) Co., Ltd. owns the source photographs. Company-authorized first author Fangjie Wu arranges their release; the repository maintainer uploads the released copies on his behalf. Upload does not transfer ownership. Image licences are assigned per file as described above; code remains Apache-2.0. Please do not attempt to identify individuals.
