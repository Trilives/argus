# Whole chain on the robot's own GPU (Todo item 1, optional): prespec

**Written:** 2026-09-27, before the FP8 model was served on the robot and before any frame
was processed in this run. **Do not edit after the run.** Outcomes go in
`../../Records/2026-09-27/`.

## Background

- **Why.** §3.2.5 and C8 state that the deployed configuration serves one 9B model in FP8 on the
  robot's 16 GB GPU for the whole chain. The user ran this and reported it on 2026-09-25, but its
  logs were not kept, and no file on the robot or on this workstation holds them. §4.9 quotes
  desktop timings only ("not robot measurements"). This run measures time and memory on the robot.
- **Checkpoint.** `Qwen3.5-9B-FP8` on the robot
  (`~/桌面/civi-site-vision-new/Model/Qwen3.5-9B-FP8`) is byte-identical to
  `CS/Model/Qwen3.5-9B-FP8`, the checkpoint of the model-swap arm Q9F: the SHA-256 of all four
  safetensors shards and `config.json` match (checked 2026-09-27).

## Fixed now

1. **Config.** `deploy/config/rcasr_runtime_v1_recal_site_q9f.json`: the site-recalibrated
   runtime config (`rcasr_runtime_v1_recal_site.json`, threshold 0.835) with only the extractor
   and the judge changed, both to `Qwen/Qwen3.5-9B-FP8`. Everything else (prior, θ, threshold,
   decoding, judge mode, §3.4 default, pins) is unchanged.
   - Why the recalibrated threshold. It is the one that reaches the target width 2 on 9B robot
     facts; the frozen threshold selected a mean width of 0.09 on this round, which would make
     judge calls, and so the measured cost, almost vanish.
2. **Serving on the robot.** The robot's own vLLM (0.21.0; the desktop runs used 0.24.0), one
   OpenAI server on port 8001 serving both roles:
   `--served-model-name Qwen/Qwen3.5-9B-FP8 --max-model-len 16384 --max-num-seqs 8
   --reasoning-parser qwen3 --gpu-memory-utilization 0.90`. The 0.90 is the documented ceiling
   on this GPU (0.95 fails at start-up, `Civi_deploy/DOG.md` §6). If the server does not start,
   the only permitted changes are `--max-num-seqs 4`, then `--max-model-len 12288`, each recorded.
   The robot's existing screening service is not running; nothing else uses the GPU.
3. **Data and run.** Round 28 + 29 (76 frames) from `~/桌面/civi-site-vision-new/Output/frames`,
   `main.py simulate --prefix dog0910f --only 28 29`, with the same runtime settings as the
   2026-09-23 runs (analysis workers 2, call workers 8, duplicate filter as configured). Frame
   hashes must match `data/field_test/manifest.json`.
4. **Measured, 1 Hz throughout the run:** `nvidia-smi` device memory used and GPU utilisation,
   and the vLLM process's own memory.
5. **Reported, whatever the values:**
   - analysis seconds per frame, p50 and p95, queue wait excluded; perception and judge parts;
   - peak and median device memory, and the vLLM process's memory, against 16,376 MiB;
   - frames analysed, errors and drops;
   - realised mean width, decision counts, judged pairs and §3.4 abstentions;
   - descriptively, per-frame agreement of selections and decisions with the 2026-09-23
     recalibrated arm (host-served 9B bf16 + 27B judge). This is not a parity test: the facts
     model's precision and the judge differ.
6. **Not claimed.** No accuracy: the round is unlabelled. No gate: this run measures cost.
7. **Into the paper** only as measured: §4.9's "desktop timings, not robot measurements"
   sentence and C8, via a revision outline first.
