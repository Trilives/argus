# On-robot round (§E6): prespec

**Written:** 2026-09-23, after the frozen-threshold run of rounds 28 and 29 on the robot,
and **before** any recalibrated threshold was applied to them. **Do not edit after the
recalibrated run.** Outcomes go in `../../Records/2026-09-23/`.

## Background

- **Why rounds 28 and 29.** The user asked to "pick a relatively good round", because no
  new site round can be counted on before submission. Rounds 28 and 29 were chosen by
  image audit and by eye, before any model output was read. They are GoPro frames of
  2026-09-10, from after the tear fix. The scene is a construction-site safety-training
  area: bright, sharp, 0 torn frames, with people both with and without hard hats and
  vests.
- **Run so far.** They ran on the robot (`deploy/`, B2, stationary in the lab) under the
  frozen config `ea624e49…`. The models are on LAN hosts reached over an SSH tunnel.
- **Why a recalibrated arm.** The realised width there was 0.09 against a target of 2.
  This is the stated facts-model difference (a 35B-fact threshold population, 9B robot
  facts). The decision already made on 2026-09-23 was to state it and recalibrate label-free.

## Fixed now

1. **Calibration population.** All *site* frames of the stored-frame pilot, and nothing
   from rounds 28 and 29:
   - local 172–176, 181, 178–180;
   - dog0904 185, 193–199, 201–204.
   These are logged score vectors under `ea624e49…`.
2. **Recalibration.** `main.py recalibrate` at target width 2. This writes
   `config/rcasr_runtime_v1_recal_site.json`. No label is read.
3. **Evaluation.** Rounds 28 and 29 are re-run on the robot under the recalibrated
   config, with the same topology and duplicate filter.
4. **Reported for both arms** (frozen and recalibrated), as proxies only:
   - decision rates;
   - realised width;
   - judged pairs and §3.4 abstentions;
   - flagged provisions;
   - latency and tokens.
5. **Not claimed.** No accuracy, precision, recall or false-alarm rate: the round is
   unlabelled. A by-eye look at flagged frames may be reported as qualitative
   illustration only.
6. **No gate.** Neither arm is declared the winner here. Which config the paper presents
   as deployed is the author's call, and both arms are reported.
