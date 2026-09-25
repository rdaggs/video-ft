# SAM 3 video-tracker fine-tuning on CVF-2026 smoke masklets

Fine-tune SAM 3's **tracker** — memory encoder + memory attention, 7.52M
parameters — so the plume/haze boundary comes from temporal state rather than
from a per-frame box. The image arm (`L4E-2`, in `sam3-finetuned-cvf`) already
won what a box can win; it bought precision with recall and has no memory.

The regime is **per-frame box prompting with prompt dropout**: a box on every
frame, matching inference, withheld on a random subset during training. Without
the dropout the shortest path to low loss runs entirely through the prompt
encoder, memory gets no gradient, and you ship a per-frame segmenter carrying a
decorative memory bank.

## Run it

```bash
./train.sh --gpu 7 --tag PhaseA > logs/$(date -u +%Y%m%d_%H%M%SZ)_PhaseA.log 2>&1 &

python -m smokeftv.config_all --check     # the snapshot against every consumer
python -m smokeftv.config_all --show      # what a run config inherits
python -m scripts.probe_masklets          # no GPU: linker diagnostics
python -m scripts.probe_clips             # no GPU: the clip corpus
python -m scripts.probe_schedule          # no GPU: dropout statistics
```

`train.sh` resolves the run name from **its own stdout** — the timestamp is
computed by the invoking shell, so it cannot recompute it without drifting by
however long startup takes. `ckpts/<run>/` and `logs/<run>.log` therefore always
share a name.

## Experiments (`/video-test`)

An experiment is run through the `/video-test` skill
(`.cursor/skills/video-test/SKILL.md`) and ends as a pushed commit adding
`experiments/<run>/`: config_resolved, the tagged log, clip manifest, label
hashes (`data.json`), checkpoint paths + sha256 (`checkpoints.json`; the `.pt`
files stay on `/data3`), the GT test-set eval, and `EXPERIMENT.md`.
`train.sh --finalize` does this when training exits 0, via
`scripts/finalize_run.py`, which runs `scripts/eval_video.py`.

* **`configs/` holds only `train_video.yaml`, edited in place.** Its git
  history is the record of previous versions; the launch commit is the code and
  config a run used. Snapshot keys are still passed as dotted overrides, never
  duplicated into the config.
* **Experiment-specific code goes in `scripts/`.** A change to `smokeftv/` must
  default to today's behaviour.
* **`eval_video.py` withholds GT from the rollout.** The in-training val pass
  does not: `mem_mask_source: supervised_best` picks the memory candidate from
  the GT mask in eval mode too, so `log.csv` val and the GT test set are not the
  same measurement. Also, `train.py` selects `best.pt` on val `iou_fused`,
  whatever `select_metric` says.

**`logs/` and `ckpts/` are symlinks into `/data3`.** The root filesystem is
100% full (22 G on 14 T). `run.feature_cache_root` points there too.

## The config contract

`training_config.yaml` is the **cross-consumer snapshot**: anything that decides
*which samples exist* or *what a metric means*. `configs/train_video.yaml` says
`extends: ../training_config.yaml` and holds only what a run varies. Precedence
is **snapshot < config file < CLI override**, and **a list is a leaf** — a
config's `val_incidents` replaces the snapshot's rather than appending, because
appending would let you only ever add to a held-out set and never state it.

`--check` fails if a key appears in **both** files. One value, one place: the
config wins the merge, so a duplicated key leaves a number in the snapshot that
nobody reads and that still looks authoritative.

`ckpts/<run>/config_resolved.yaml` strips `extends` and is fully literal. It is
the file you reach for to reproduce a run, and it would not be a record of
anything if its values could move when the snapshot is next edited.

**`divergences:` records disagreements on purpose, and `--check` asserts each is
STILL TRUE.** Quietly fixing one without deleting its entry is a failure. There
are eleven; the load-bearing ones are `no_two_plume`, `memory_conditioning`,
`bbox_repair_for_baseline` and `fuse_fill_frac`.

## Things that will cost you a day

**`build_tracker` builds the architecture and nothing else.** Without
`model.load_base_weights` the tracker is randomly initialised everywhere
L4E-2's 66 tensors do not reach. It does not crash. It produces near-full-frame
masks with `iou_head` ~0.5 and `object_score_logits` ~0 — which reads as "the
memory path does not work yet", not as "there are no weights". This cost a real
debugging cycle; `load_base_weights` now refuses to return if any parameter got
no released weights.

**`Sam3TrackerPredictor.__init__` enters a process-global bf16 autocast and
never exits it** (`sam3_tracking_predictor.py:50-51`), with `cache_enabled=True`.
That cache is keyed by tensor identity and is not invalidated by an in-place
`optimizer.step()`, so the forward will happily serve pre-update weights. No
crash, real numbers, wrong label. `build_tracker` exits it; `engine.autocast_for`
always passes `cache_enabled=False`.

**The training loop drives `track_step` directly and owns `output_dict`.** Every
`Sam3TrackerPredictor` entry point is decorated `@torch.inference_mode()`, so
its tensors can never enter autograd. `propagate_in_video_preflight` also
re-files outputs by storage key, and `clear_non_cond_mem_around_input=True`
*deletes the recency memory around every prompted frame* — the exact opposite of
this experiment.

**`prompted_frames_are_conditioning: false` is one line in `rollout.py`:**
frame 0 and explicit re-inits go to `cond_frame_outputs`, everything else to
`non_cond_frame_outputs` regardless of whether it carried a prompt. Upstream has
it the other way (`:264-266`, flag hardcoded True at `:54`), which under
per-frame prompting makes EVERY frame conditioning: the cap saturates, the FIFO
is evicted, and `:673-678` forces `t = 0` so the frame loses its temporal
identity too. The bank composition table on batch 0 is how you check this; if
`n_cond` climbs past 1, stop.

**A box prompt disables multimask.** `multimask_min_pt_num=0, max_pt_num=1` and
a box is two points, so `_use_multimask` returns False on prompted frames and
True on dropout frames — one candidate against three, with different
object-pointer token semantics, inside one clip. `model.multimask_mode: always`
forces three everywhere.

**`.train()` is required but is not enough.** It raises `AttributeError` twice
on attributes `__init__` never sets (`:333`, `:684`), and switches on 0.1
dropout across all four memory-attention layers. What it buys is `:821`: in eval
mode the mask is hard-binarised before the memory encoder — zero gradient — and
in train mode it is a differentiable sigmoid. Our regime prompts most frames, so
eval mode would starve `maskmem_backbone` almost entirely.

**`object_score_logits <= 0` silently zeroes a frame.** The mask is replaced by a
constant `NO_OBJ_SCORE`, so the frame contributes exactly zero gradient,
`obj_ptr` collapses to `no_obj_ptr`, and that frame's memory is poisoned. It
reads as "the loss plateaued". `log.csv` carries `zero_grad_frames` per epoch;
`model.force_obj_appearing: true` teacher-forces presence from the GT in Phase A.

**`vitdet.Mlp` routes through `perflib.fused.addmm_act`**, which raises on the
*global* grad flag and casts to bf16 unconditionally while `fc2` respects
autocast. So the backbone must run under the same autocast as the rollout, and
`model.patch_vit_mlp` must land before Phase C.

**Clip frame indices are relabelled contiguous `0..L-1`.** The recency FIFO looks
up `non_cond_frame_outputs[frame_idx - t_rel]`, so a strided set of real frame
indices silently produces an EMPTY memory bank with no error at all.

## What the data does not contain

* **No two-plume ground truth.** `annotations.json` has <=1 box per frame in 519
  of 529 incidents; the multi-instance polygon frames are wisps of one plume
  (hence `gt_merge: union`); `1173104_ec-395-2`, whose `status.json` says
  "segmented 2x smoke plume well", has one box and one polygon per frame. So
  `max_masklets_per_clip` is 1 and the identity metrics are out of scope.
* **No detector boxes.** Every `annotations.json` is labelbox GT.
  `sam3_pred.coco.json` is not a substitute — `prompt_mode: pvs` means SAM 3
  prompted by the GT box.
* **No smoke-base labels on the training corpus.** They exist for 26 of the 28
  incidents under `splits.gt_dataset_root`, i.e. the held-out test set only.

## Conventions

* Comments explain **why**, not what. The upstream poc is written that way.
* A checkpoint stores only `requires_grad` tensors — 30 MB, not 3.4 GB.
  `last_resume.pt` is a **separate artifact** carrying the optimizer and five RNG
  streams; deleting it does not touch a scored checkpoint.
* Metrics go to `log.csv` (flat, one row per epoch per split) and
  `metrics.jsonl` (nested: `iou_by_t`, `iou_by_kind`). **Do not add
  TensorBoard.** `iou_by_t` is the one to read — a memory bug is a monotone
  decay in `t` that a clip mean hides completely.
* `lr_schedule` is stepped **per optimizer step**, and `lr_warmup_frac` is a
  fraction of total steps, so changing `epochs` does not silently change warmup.
* `train.clip_policy: sequential` does not scale — the image repo measured val
  IoU 0.5569 -> 0.4548 in one epoch under its equivalent.
* Every port is differential-tested against `sam3-finetuned-cvf` rather than
  eyeballed. If you touch `crops`, `losses`, `metrics` or `postprocess`, re-run
  those tests: behavioural equality is what makes any cross-repo number mean
  anything.

## Not done yet

* **The arm-1 acceptance gate.** `iou_polygon` on `L4E-2` must reproduce
  **0.6620960408** over 1412 capped samples / 27 incidents, with
  `prompting.bbox_repair.enabled: true` — without that flag the same weights give
  0.6577468796, and 0.0044 is the same size as a real port bug. Until this
  passes, a video number cannot be compared to the image arm.
* The feature cache (`features.py`) and Phase B/C. `eval_video.py` scores the
  three inference modes on GT clips, but its numbers are not comparable to the
  image arm until the gate above passes.
