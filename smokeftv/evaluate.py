"""Scoring a tracker on the GT test set, one row per (clip, t, inference mode).

The unit is the training clip, not the frame, because the question the video
arm asks is positional: does the mask at t=7 hold up as well as at t=1? A
memory bug is a monotone decay in `t` that a frame mean hides completely, so
every row carries its clip position and the report is built around `iou_by_t`.

Clips are built by `corpus.build_clips` from the run's OWN config pointed at
`splits.gt_dataset_root`, with `stride_jitter: 0` and `reverse_prob: 0` so the
list is a pure function of (config, data). Same length, same realised stride,
same window rule as training, so `t` means here what it meant in `log.csv`.

Nothing at test time sees GT except the scorer. `targets_high` is withheld from
the rollout (see `rollout_clip`), the tracker is in eval mode so presence comes
from the head rather than teacher forcing, and the memory candidate is the IoU
head's pick. The in-training val pass is not that strict, so the two numbers
are not interchangeable.

Three modes, the snapshot's `eval_video.inference_modes`:

  prompt_every_frame  a box on every frame. The per-frame regime with memory on.
  keyframe            a box every `keyframe_every` frames; at L=8 that is t=0
                      only, so t1..t7 are pure propagation — the memory test.
  conditional         propagate, and re-prompt when the propagated mask's bbox
                      disagrees with the frame's box below
                      `conditional_reprompt_iou`. The shippable arm. Re-prompts
                      are counted; that count is `restarts`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from smokeftv.corpus import build_clips
from smokeftv.dataset import ClipDataset, collate
from smokeftv.engine import _prepare_feats, autocast_for
from smokeftv.incidents import camera_of, discover_incidents, load_polygons
from smokeftv.metrics import fuse_soft, iou_from_masks
from smokeftv.postprocess import (PocParams, _counts, fused_surface,
                                  polygon_surface, upsample_logits)
from smokeftv.rollout import rollout_clip

log = logging.getLogger("smokeftv")

MODES = ("prompt_every_frame", "keyframe", "conditional")

# The rollout's decoder output grid. A propagated mask's bbox is measured here
# and scaled up to image_size, where the prompt boxes live.
LOW_RES = 288


@dataclass(frozen=True)
class EvalParams:
    """The `eval_video:` knobs a row depends on."""

    modes: tuple[str, ...] = MODES
    keyframe_every: int = 8
    conditional_reprompt_iou: float = 0.5


def eval_overrides(gt_root: str, box_jitter: tuple[float, float] | None = None
                   ) -> list[str]:
    """What turns a run's training config into its test-set config, and nothing
    more: the data root, the two clip knobs that are random draws, and the box
    jitter — off unless asked for, and then the SAME band for every run, so a
    run trained with jitter is not scored on its own augmentation."""
    jitter = (["prompt.box_jitter.enabled=false"] if box_jitter is None else
              ["prompt.box_jitter.enabled=true", "prompt.box_jitter.seed=0",
               f"prompt.box_jitter.pad_min={box_jitter[0]}",
               f"prompt.box_jitter.pad_max={box_jitter[1]}"])
    return [f"data.root={gt_root}", "data.clips.stride_jitter=0",
            "data.clips.reverse_prob=0.0", *jitter]


def eval_clips(cfg):
    incidents = discover_incidents(cfg.data.root)
    clips, reports = build_clips(cfg, incidents, "test")
    return incidents, clips, reports


def sample_id(clip_id: str, t: int) -> str:
    return f"{clip_id}:t{t}"


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def _mask_box(mask: torch.Tensor, scale: float):
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        return None
    return (float(xs.min()) * scale, float(ys.min()) * scale,
            float(xs.max() + 1) * scale, float(ys.max() + 1) * scale)


def _policy(mode: str, length: int, params: EvalParams, boxes, image_size: int,
            poc: PocParams):
    """`(keep_mask, reprompt)` for one clip under one mode."""
    if mode == "prompt_every_frame":
        return [True] * length, None
    keep = [t % params.keyframe_every == 0 for t in range(length)]
    if mode == "keyframe":
        return keep, None
    if mode != "conditional":
        raise ValueError(f"unknown inference mode {mode!r}; known are {MODES}")

    scale = image_size / LOW_RES

    def reprompt(t: int, out) -> bool:
        # The same fusion the product applies, so the decision is made on the
        # mask a user would have seen, not on the IoU head's single pick.
        fused = fuse_soft(out.low_res_multimasks.float(), out.ious.float(),
                          top_k=poc.top_k, max_fill_frac=poc.max_fill_frac)[0]
        box = _mask_box(fused, scale)
        if box is None:
            return True
        return _box_iou(box, boxes[t].tolist()) < params.conditional_reprompt_iou

    return keep, reprompt


class _FeatCache:
    """Backbone features per (masklet window, frame).

    Clips of one masklet overlap heavily — 449 GT clips cover 808 unique frames —
    and share one crop window, so a frame's features are identical in every clip
    that contains it. Cleared when the masklet changes, which bounds it at one
    masklet's frames (~10 MiB each).
    """

    def __init__(self) -> None:
        self.owner = None
        self.feats: dict[int, tuple] = {}

    def get(self, tracker, cfg, clip, images):
        owner = (clip.incident, clip.masklet_id, clip.window)
        if owner != self.owner:
            self.owner, self.feats = owner, {}
        missing = [i for i, f in enumerate(clip.frame_indices) if f not in self.feats]
        if missing:
            for i, feat in zip(missing, _prepare_feats(tracker, images[missing], cfg)):
                self.feats[clip.frame_indices[i]] = feat
        return [self.feats[f] for f in clip.frame_indices]


def score_clips(tracker, clips, cfg, *, params: EvalParams, poc: PocParams,
                geometry, limit: int | None = None, log_every: int = 25
                ) -> list[dict]:
    """Every (clip, t, mode) row for one set of weights."""
    clips = list(clips)[:limit] if limit else list(clips)
    loader = DataLoader(ClipDataset(clips, cfg, "test"), batch_size=1,
                        shuffle=False, num_workers=cfg.train.num_workers,
                        collate_fn=collate)
    cache = _FeatCache()
    polygons: dict[str, tuple] = {}
    loss_size = cfg.train.loss_size
    rows: list[dict] = []

    for step, batch in enumerate(loader):
        clip = clips[batch["index"]]
        if clip.incident not in polygons:
            polygons[clip.incident] = load_polygons(cfg.data.root, clip.incident)
        gt_polys, _ = polygons[clip.incident]
        length = len(clip.frame_indices)

        batch["feats"] = cache.get(tracker, cfg, clip, batch["images"])
        batch["boxes"] = batch["boxes"].to(cfg.device)
        targets = batch["targets"].to(cfg.device)
        batch["targets_high"] = None

        for mode in params.modes:
            keep, reprompt = _policy(mode, length, params, batch["boxes"],
                                     cfg.crop.image_size, poc)
            with torch.no_grad(), autocast_for(cfg):
                outs, labels = rollout_clip(tracker, batch, cfg, keep_mask=keep,
                                            reprompt=reprompt)
            previous_pred = previous_gt = None
            for t, (out, label) in enumerate(zip(outs, labels)):
                logits = out.low_res_multimasks.float()
                ious = out.ious.float()
                target = targets[t][None]
                fused = fused_surface(logits, ious, target, loss_size, poc)
                pred = fuse_soft(upsample_logits(logits, (loss_size, loss_size)),
                                 ious, top_k=poc.top_k,
                                 max_fill_frac=poc.max_fill_frac)
                frame = clip.frame_indices[t]
                poly = polygon_surface(logits[0], ious[0], clip.window,
                                       gt_polys.get(frame, []), clip.image_hw,
                                       poc, geometry)
                counts = _counts(int(fused["pred_px"][0]), int(fused["gt_px"][0]),
                                 int(fused["inter_px"][0]))
                gt_now = target > 0.5
                kind = ("cond" if t == 0 else
                        "prompted" if label["prompted"] else "propagated")
                rows.append({
                    "sample_id": sample_id(clip.clip_id, t),
                    "mode": mode,
                    "incident": clip.incident,
                    "camera": camera_of(clip.incident),
                    "masklet_id": clip.masklet_id,
                    "clip_id": clip.clip_id,
                    "t": t,
                    "frame_index": frame,
                    "stride": clip.stride,
                    "kind": kind,
                    "prompted": int(label["prompted"]),
                    "reprompted": int(label["reprompted"]),
                    "box_gap": int(batch["box_gap"][t]),
                    "obj_score": round(float(out.object_score_logits.float().min()), 4),
                    "iou_fused": round(float(fused["iou_fused"][0]), 6),
                    "iou_selected": round(float(fused["iou_selected"][0]), 6),
                    "iou_best": round(float(fused["iou_best"][0]), 6),
                    "prec_fused": round(counts["precision"], 6),
                    "recall_fused": round(counts["recall"], 6),
                    "gt_px_fused": int(fused["gt_px"][0]),
                    "pred_px_fused": int(fused["pred_px"][0]),
                    "iou_polygon": round(poly["iou"], 6),
                    "prec_polygon": round(poly["precision"], 6),
                    "recall_polygon": round(poly["recall"], 6),
                    "gt_px_polygon": poly["gt_px"],
                    "pred_px_polygon": poly["pred_px"],
                    "gt_px_outside_crop": poly["gt_px_outside_crop"],
                    # Frame-to-frame agreement of the prediction, beside the same
                    # number for the GT: flicker is the gap between the two, and
                    # a model that freezes its first mask beats the GT on this
                    # column alone, so it is never read without iou_by_t.
                    "continuity": ("" if previous_pred is None else
                                   round(float(iou_from_masks(pred, previous_pred)[0]), 6)),
                    "gt_continuity": ("" if previous_gt is None else
                                      round(float(iou_from_masks(gt_now, previous_gt)[0]), 6)),
                })
                previous_pred, previous_gt = pred, gt_now

        if log_every and step % log_every == 0:
            log.info("  [%4d/%d] %s", step, len(clips), clip.clip_id)
    return rows
