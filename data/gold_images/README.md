# Gold images (v2.2.0)

The 500 photographs of the frozen gold set, with workers' faces blurred. Their labels are the
label-only gold in [`../../reproduce/gold_labels.json`](../../reproduce/gold_labels.json), keyed by
the same `img_NNNN` ids; the reliability report is in `../../reproduce/reliability_agreement.json`.

- **Anonymisation.** Each visible face is covered by a tight box over the eyes, nose and mouth and
  Gaussian-blurred, so helmets, vests and harnesses (what the rules check) stay visible. Every image
  was reviewed by a person, who drew the boxes; machine proposals were only drafts. A person
  detector checked the result: each person it found has a face box or a recorded reason the face is
  not identifiable. A printed notice with a mobile number and four company names are also blurred.
  The pixel checks (residual detail in each box, no change outside the boxes) are recorded per image
  in `anonymisation.json`.
- **Format.** JPEG, longest side at most 2048 px, re-encoded without EXIF or other metadata.
  `manifest.csv` lists each file's SHA-256 and size.
- **Not the scored pixels.** The reported results were computed on the original photographs; blurring
  and scaling change pixels, so re-running the models on these files will not reproduce every number
  exactly. The label-only gold and the released predictions reproduce the scores themselves.
- **Sites.** 346 images belong to the 48 sites of the internal evaluation; 154 have no recoverable
  site and are not used in any RCASR result. Site keys are not released.
- **Licence.** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see `LICENSE`). The
  photographs are operational inspection records of China Construction Third Engineering Bureau
  (Shenzhen) Co., Ltd. Do not attempt to identify the people shown.
