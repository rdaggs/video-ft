"""The training and evaluation loops.

Three placement rules that are correctness, not tuning:

* **autocast wraps the rollout only, with `cache_enabled=False`.** The cache is
  keyed by tensor identity and is not invalidated by an in-place
  `optimizer.step()`, so with it on the forward can serve pre-update weights.
* **the loss is computed outside autocast on `.float()` logits.** Dice sums over
  ~10^6 pixels and bf16's mantissa destroys real gradient in that sum.
* **the clip is a recurrence, not 8 independent frames.** Frame t's memory
  attention consumes frames t-1..t-6, each tracing back through its own memory
  encoder, so activation memory sums over the whole clip.
"""

from __future__ import annotations

import functools
import logging
import math
import time

import torch

from smokeftv.losses import MaskLoss
from smokeftv.metrics import all_ious
from smokeftv.postprocess import upsample_logits
from smokeftv.prompting import frame_keep_mask
from smokeftv.rollout import BankProbe, rollout_clip

log = logging.getLogger("smokeftv")


def autocast_for(cfg):
    if cfg.train.amp_dtype == "float32" or cfg.device != "cuda":
        return torch.autocast("cuda", enabled=False)
    dtype = torch.bfloat16 if cfg.train.amp_dtype == "bfloat16" else torch.float16
    # cache_enabled=False is mandatory, not a perf tweak. See the module docstring.
    return torch.autocast("cuda", dtype=dtype, cache_enabled=False)


def frame_weight(cfg, t: int, prompted: bool) -> float:
    """Frame 0 is conditioning and trivially easy — it has a box and no memory
    to get wrong. Dropout frames are the only ones measuring whether memory
    works, so they are weighted separately even though the default is 1.0."""
    loss = cfg.train.loss
    if t == 0:
        return loss.cond_frame_weight
    return loss.prompted_frame_weight if prompted else loss.dropout_frame_weight


def clip_loss(cfg, criterion: MaskLoss, outs, batch, keep_mask):
    """MaskLoss per frame, weighted, plus the per-position breakdown.

    The breakdown matters more than the mean: a memory bug shows up as a
    monotone IoU decay in `t`, and a clip mean hides it completely.
    """
    loss_size = cfg.train.loss_size
    total = torch.zeros((), device=outs[0].low_res_multimasks.device)
    weight_sum = 0.0
    per_t, per_kind = [], {"cond": [], "prompted": [], "dropout": []}
    oracle = {"iou_best": 0.0, "iou_selected": 0.0}
    zero_grad_frames = 0

    for t, out in enumerate(outs):
        target = batch["targets"][t].unsqueeze(0)
        # .float() and outside autocast: see the module docstring.
        logits = upsample_logits(out.low_res_multimasks.float(),
                                 (loss_size, loss_size))
        result = criterion(logits, out.ious.float(), target)
        weight = frame_weight(cfg, t, bool(keep_mask[t]))
        total = total + weight * result.total
        weight_sum += weight
        # Not `result.iou`: that is the GT-picked best-of-3 candidate, an oracle
        # no inference path can reach, and it read ~0.07 above the fused mask
        # the poc actually ships. `iou_fused` is also what fused_surface scores
        # at eval, so the two logs are the same number by construction.
        with torch.no_grad():
            ious = all_ious(logits.detach(), out.ious.detach().float(), target)
        iou = float(ious["iou_fused"].mean())
        for key in oracle:
            oracle[key] += float(ious[key].mean()) / len(outs)
        per_t.append(iou)
        kind = "cond" if t == 0 else ("prompted" if keep_mask[t] else "dropout")
        per_kind[kind].append(iou)
        if float(out.object_score_logits.detach().min()) <= 0:
            zero_grad_frames += 1

    return total / max(weight_sum, 1e-8), {
        "iou_by_t": per_t,
        "iou_by_kind": {k: (sum(v) / len(v) if v else float("nan"))
                        for k, v in per_kind.items()},
        **oracle,
        "zero_grad_frames": zero_grad_frames,
    }


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """Stepped per OPTIMIZER STEP, not per epoch, and `lr_warmup_frac` is a
    fraction of total steps — so changing `epochs` does not silently change the
    warmup."""
    if cfg.train.lr_schedule == "none":
        return None
    total = max(steps_per_epoch * cfg.train.epochs, 1)
    warmup = int(cfg.train.lr_warmup_frac * total)
    floor = cfg.train.lr_min_factor

    def curve(step: int) -> float:
        if warmup and step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, curve)


def _prepare_feats(tracker, images, cfg):
    """Backbone once per frame, under no_grad — it is frozen in Phase A.

    Under the SAME autocast as the rollout, not bare fp32: `vitdet.Mlp` routes
    through `perflib.fused.addmm_act`, which casts to bf16 unconditionally while
    `fc2` respects autocast, so fp32 here raises "mat1 and mat2 must have the
    same dtype" inside the trunk.
    """
    out = []
    with torch.no_grad(), autocast_for(cfg):
        for t in range(images.shape[0]):
            backbone_out = tracker.forward_image(images[t:t + 1].to(cfg.device))
            _, feats, pos, sizes = tracker._prepare_backbone_features(backbone_out)
            out.append((feats, pos, sizes))
    return out


@functools.lru_cache(maxsize=1)
def _eval_params():
    from smokeftv import config_all
    from smokeftv.evaluate import EvalParams
    ev = config_all.load()["eval_video"]
    return EvalParams(keyframe_every=int(ev.get("keyframe_every", 8)),
                      conditional_reprompt_iou=float(ev.get("conditional_reprompt_iou", 0.5)))


def _val_policy(cfg, length: int, boxes):
    from smokeftv.evaluate import _policy
    from smokeftv.postprocess import PocParams
    return _policy(cfg.train.inference_mode, length, _eval_params(), boxes,
                   cfg.crop.image_size, PocParams())


def run_epoch(tracker, loader, cfg, *, epoch: int, optimizer=None, scheduler=None,
              criterion=None, log_every: int = 50, train: bool = True,
              probe_first: bool = False):
    """One pass. `optimizer is None` means evaluation."""
    device = cfg.device
    criterion = criterion or MaskLoss(cfg.train.loss)
    totals = {"loss": 0.0, "iou": 0.0, "iou_best": 0.0, "iou_selected": 0.0,
              "prompted": 0.0, "n": 0, "zero_grad_frames": 0}
    by_t: list[float] = []
    by_kind = {"cond": 0.0, "prompted": 0.0, "dropout": 0.0}
    kind_n = {"cond": 0, "prompted": 0, "dropout": 0}
    started = time.time()
    probe = None

    for step, batch in enumerate(loader):
        keep_mask, p = frame_keep_mask(
            len(batch["frame_indices"]) if "frame_indices" in batch
            else batch["images"].shape[0],
            cfg.train.dropout, epoch, cfg.data.clips.seed, batch["clip_id"])
        batch["feats"] = _prepare_feats(tracker, batch["images"], cfg)
        batch["boxes"] = batch["boxes"].to(device)
        batch["targets"] = batch["targets"].to(device)
        reprompt = None
        if train:
            # (1, 1, S, S) — the batch dim is kept: _forward_sam_heads flattens
            # from dim 1 for the presence test, and track_step_train expands it
            # across the candidate axis for the memory-mask selection.
            batch["targets_high"] = [
                torch.nn.functional.interpolate(
                    batch["targets"][t][None, None], size=(cfg.crop.image_size,) * 2,
                    mode="nearest") for t in range(batch["targets"].shape[0])]
        else:
            # Val is inference: train.inference_mode's prompting policy, taken
            # from the same code eval_video scores with, and no GT inside the
            # rollout — with it, supervised_best picks the memory candidate
            # from the answer and force_obj_appearing hides a presence miss.
            keep_mask, reprompt = _val_policy(cfg, len(keep_mask), batch["boxes"])
            batch["targets_high"] = None

        if probe_first and step == 0:
            probe = BankProbe().attach(tracker)

        with autocast_for(cfg):
            outs, labels = rollout_clip(tracker, batch, cfg, keep_mask=keep_mask,
                                        reprompt=reprompt)
        # What actually carried a box, conditional re-prompts included.
        keep_mask = [label["prompted"] for label in labels]
        loss, stats = clip_loss(cfg, criterion, outs, batch, keep_mask)

        if probe is not None and step == 0:
            probe.detach()
            _log_bank(probe, labels)
            probe = None

        if train:
            (loss / cfg.train.grad_accum).backward()
            if (step + 1) % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    cfg.train.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        totals["loss"] += float(loss.detach())
        totals["iou"] += sum(stats["iou_by_t"]) / len(stats["iou_by_t"])
        totals["iou_best"] += stats["iou_best"]
        totals["iou_selected"] += stats["iou_selected"]
        totals["prompted"] += sum(keep_mask) / len(keep_mask)
        totals["zero_grad_frames"] += stats["zero_grad_frames"]
        totals["n"] += 1
        if not by_t:
            by_t = [0.0] * len(stats["iou_by_t"])
        for i, value in enumerate(stats["iou_by_t"]):
            by_t[i] += value
        for kind, value in stats["iou_by_kind"].items():
            if value == value:                       # not NaN
                by_kind[kind] += value
                kind_n[kind] += 1

        if log_every and step % log_every == 0:
            log.info("  %s e%d [%4d/%d] loss %.4f iou %.4f p=%.2f",
                     "train" if train else "val  ", epoch, step, len(loader),
                     float(loss), sum(stats["iou_by_t"]) / len(stats["iou_by_t"]), p)

    n = max(totals["n"], 1)
    return {
        "loss": totals["loss"] / n,
        "iou": totals["iou"] / n,
        "iou_fused": totals["iou"] / n,
        "iou_best": totals["iou_best"] / n,
        "iou_selected": totals["iou_selected"] / n,
        "prompted_frac": totals["prompted"] / n,
        "iou_by_t": [v / n for v in by_t],
        "iou_by_kind": {k: (by_kind[k] / kind_n[k] if kind_n[k] else float("nan"))
                        for k in by_kind},
        "zero_grad_frames": totals["zero_grad_frames"],
        "clips": n,
        "secs": round(time.time() - started, 1),
    }


def _log_bank(probe: BankProbe, labels) -> None:
    """Bank composition on batch 0 of every run, as the handoff requires.

    Measured off the tensor memory attention actually received, then checked
    against what the rollout believes it filed. A mismatch here is a silent
    reclassification, which is otherwise invisible until the metrics move.
    """
    log.info("  memory bank composition (batch 0):")
    log.info("    %3s %6s %9s | %9s %9s | %9s", "t", "cond", "prompted",
             "n_cond", "n_recent", "attended")
    for label, row in zip(labels, probe.rows):
        log.info("    %3d %6s %9s | %9d %9d | %9d",
                 label["t"], label["is_cond"], label["prompted"],
                 label["n_cond"], label["n_recent"], row["mem_frames"])
    for label, row in zip(labels[1:], probe.rows[1:]):
        expect = min(label["n_cond"] + label["n_recent"], row["mem_frames"])
        if row["mem_frames"] != expect:
            log.info("    WARNING t=%d: attended %d memory frames, the rollout "
                     "filed %d cond + %d recent. A silent reclassification.",
                     label["t"], row["mem_frames"], label["n_cond"],
                     label["n_recent"])
