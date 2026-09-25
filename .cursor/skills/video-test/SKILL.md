---
name: video-test
description: Implement, launch and record one SAM 3 video-tracker experiment in video-ft so it can be recreated exactly. Use when the user invokes /video-test with an experiment to run.
disable-model-invocation: true
---

# /video-test — one experiment, fully recorded

The user writes: `/video-test implement {experiment}`. Implement it, launch it,
and make sure that when it finishes the repo holds everything needed to
recreate it. Read `CLAUDE.md` first; its traps apply to every experiment.

## The contract

Every experiment ends as one git commit, pushed, that adds `experiments/<run>/`:

| What | Where it lives |
|---|---|
| code the run used | the **launch commit** (`provenance.json` git.sha); `train.sh` refuses a dirty tree |
| config | `configs/train_video.yaml` at the launch commit, plus the literal merged `config_resolved.yaml` |
| log, tagged | `experiments/<run>/<timestamp>_<tag>.log` |
| training data manifest | `clip_manifest.jsonl`, `splits_resolved.yaml`, `data.json` (per-incident label hashes) |
| checkpoints | stay on `/data3`; `checkpoints.json` records path + sha256 |
| GT test-set eval | `gt_test_set/` (report.txt, per-arm frames.csv / incidents.csv / summary.json) |
| summary | `EXPERIMENT.md`: intent, results, reproduce recipe |

`train.sh --finalize` produces all of it automatically when training exits 0
(`scripts/finalize_run.py`, which runs `scripts/eval_video.py`). You do not
wait for the run.

## Rules

- **`configs/` holds only `train_video.yaml`.** Edit it in place. Previous
  versions are its git history, so never add per-experiment config files.
- **Experiment-specific code goes in `scripts/`**, e.g. `scripts/exp_<tag>.py`.
  A change to `smokeftv/` must be behind a config flag that defaults to today's
  behaviour, because every earlier record assumes that behaviour.
- **Snapshot keys** (anything `python -m smokeftv.config_all --show` lists as
  inherited from `training_config.yaml`) are not edited for one experiment. Pass
  them as dotted overrides on the `train.sh` line. They land in
  `provenance.json` and `config_resolved.yaml`. `--check` fails if a key is in
  both files.
- If you touch `crops`, `losses`, `metrics` or `postprocess`, re-run their
  differential tests (`CLAUDE.md`, Conventions).

## Workflow

1. **Pin down the experiment.** Write down the hypothesis, the one thing that
   changes, what result would confirm or refute it, and a short `tag`
   (`[A-Za-z0-9._-]`). If the change is ambiguous, ask before editing.
2. **Start clean:** `git status` should be clean on `main`; then `git pull`.
   Do not commit unrelated dirty work; ask the user about it.
3. **Implement.** Edit `configs/train_video.yaml` and/or add `scripts/exp_<tag>.py`.
   Keep the YAML's why-comments true for the new values.
4. **Validate, no GPU:**
   ```bash
   .venv/bin/python -m smokeftv.config_all --check
   .venv/bin/python -m scripts.train --config configs/train_video.yaml \
     --run-name _dry_<tag> --ckpt-dir ckpts/_dry_<tag> --dry-run [overrides]
   rm -rf ckpts/_dry_<tag>
   ```
   For dropout changes, also run `python -m scripts.probe_schedule`.
5. **Launch commit**, then push. The body is the record's "Intent" section:
   ```
   exp(<tag>): <one line>

   Hypothesis: ...
   Change: ...
   Success: ...
   Command: ./train.sh --gpu <N> --tag <tag> --finalize [overrides]
   ```
6. **Pick a GPU.** Use the one the user named. Otherwise pick one that
   `nvidia-smi` shows idle, and say which one you chose.
7. **Launch:**
   ```bash
   ./train.sh --gpu <N> --tag <tag> --finalize [overrides] \
     > logs/$(date -u +%Y%m%d_%H%M%SZ)_<tag>.log 2>&1 &
   ```
   Wait until the log shows the banner and `memory bank composition`. If
   `n_cond` climbs past 1, stop the run (`CLAUDE.md`).
8. **Report to the user:** run name, GPU, PID, log path, the launch commit, and
   that `results(<tag>): <run>` will be pushed when it finishes.

## Experiments without training

If the experiment only evaluates, for example another checkpoint or inference
mode:

- Write it as `scripts/exp_<tag>.py`.
- Put its outputs in `ckpts/<timestamp>_<tag>/`, including a `provenance.json`
  with `tag`, `git.sha` and `status`.
- Commit the script, run it, then record it:
  `python -m scripts.finalize_run --run <timestamp>_<tag>`.
  The finalizer copies anything in the run directory and scores the GT set only
  when there is a checkpoint and a `config_resolved.yaml`.

## When something fails

- **Training exits non-zero:** `train.sh` does not finalize. Once the cause is
  understood, record it anyway:
  `.venv/bin/python -m scripts.finalize_run --run <run>`.
- **Eval fails:** the record is still committed, with the GT test set marked
  `failed`. Rerun by hand:
  `CUDA_VISIBLE_DEVICES=<N> .venv/bin/python -m scripts.finalize_run --run <run> --refresh-eval`
- **Push fails:** the commit stays local. Run `git pull --rebase && git push`.

## Reading a result

Read `iou_by_t` first. A memory bug shows up as a monotone decay in `t`.
`keyframe` mode is the pure-memory test, because t1..t7 are unprompted.
`conditional` is the shippable arm. `init` is the weights the run started from,
so the delta against it is what the memory training bought. GT test numbers are
not comparable to the image arm's 0.662 until the arm-1 acceptance gate passes.
