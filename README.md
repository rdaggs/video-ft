# sam3-video-cvf

Fine-tune SAM 3's **video tracker** — the memory encoder and memory attention —
on hand-corrected CVF-2026 smoke masklets, so the boundary between plume and
haze comes from temporal state rather than from a per-frame box.

Today's shipped inference is detection-conditioned per-frame segmentation: every
frame gets a box, SAM 3 image mode returns three candidates, they are soft-fused
and turned into a polygon. The image finetune (`L4E-2`) is a large win on that
surface, but it has no temporal state, and the recall it traded away is exactly
the boundary a box cannot describe.

The training regime is **per-frame box prompting with prompt dropout**: a box is
available on every frame, matching inference, but it is withheld on a random
subset of frames so memory attention is load-bearing. Without the dropout the
shortest path to low loss runs entirely through the prompt encoder, memory gets
no gradient, and you ship a per-frame segmenter carrying a decorative memory
bank.

## Run it

```bash
uv sync
./train.sh --gpu 7 --tag PhaseA > logs/$(date -u +%Y%m%d_%H%M%SZ)_PhaseA.log 2>&1 &
```

`train.sh` resolves the run name from its own stdout, so `ckpts/<run>/` and
`logs/<run>.log` always share a name. `logs/` and `ckpts/` are symlinks onto
`/data3` because the root filesystem is full.

With `--finalize`, a run that exits 0 scores the GT test set and commits and
pushes `experiments/<run>/`, which holds the record needed to recreate it. See
`.cursor/skills/video-test/SKILL.md`.

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/eval_video.py --run ckpts/<run>
.venv/bin/python -m scripts.finalize_run --run <run>    # record a run by hand
```

```bash
python -m smokeftv.config_all --check   # the snapshot against every consumer
python -m smokeftv.config_all --show    # what a run config inherits
python -m scripts.probe_masklets        # no GPU: linker diagnostics
python -m scripts.probe_clips           # no GPU: the clip corpus
```

## The contract

`training_config.yaml` is the **cross-consumer snapshot**: everything that
decides *which samples exist* or *what a metric means*. `configs/train_video.yaml`
says `extends: ../training_config.yaml` and holds only what a run varies — lr,
epochs, the freeze schedule, the dropout distribution. No key appears in both,
and `--check` fails if one does.

`ckpts/<run>/config_resolved.yaml` strips `extends` and is a self-contained
literal record. Reproducing a run means reaching for that file, and it would not
be a record of anything if its values could move when the snapshot is next
edited.

`divergences:` records disagreements on purpose. `--check` asserts each is
**still true**, so quietly fixing one without deleting its entry is a failure.

See `CLAUDE.md` for the details that cost real debugging time.
