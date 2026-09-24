"""The mask loss. Ported verbatim from `sam3-finetuned-cvf/smokeft/losses.py`.

Behavioural equality with that file is the acceptance criterion: `iou_polygon`
here and `iou_polygon` there have to mean the same thing or every comparison
against the image arm is void. The only edit is the config field names
(`dice_weight` / `bce_weight` / `iou_head_weight` rather than `dice` / `bce` /
`iou`), because this repo's `LossConfig` also carries the video terms.

The weights stay 1.0/1.0/1.0 — L4E-2's recipe, off its own
`config_resolved.yaml`. An earlier draft of the snapshot carried SAM 2's video
recipe (focal 20.0); nothing that produced a recorded number used it, and the
loss SCALE is what makes `train.lr` transfer between the two arms.


Step 1 keeps it to what SAM itself trains with, minus the extras: Dice + BCE on
the mask, MSE on the IoU-prediction head, and — because the decoder returns three
ambiguity candidates — supervision applied only to the candidate that best matches
the target. That last part matters: averaging the loss over all three collapses
them toward each other and destroys the diversity the poc's top-3 soft fusion
relies on.

The boundary-aware term from the plan is a later step and is deliberately absent.
When it lands it goes here, as another weighted term in `MaskLoss.forward`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from smokeftv.config import LossConfig


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice per sample. `logits` and `target` are (N, H, W)."""
    probs = logits.sigmoid().flatten(1)
    flat = target.flatten(1)
    numerator = 2 * (probs * flat).sum(-1)
    denominator = probs.sum(-1) + flat.sum(-1)
    return 1 - (numerator + eps) / (denominator + eps)


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample mean binary cross-entropy. `logits`/`target` are (N, H, W)."""
    return F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    ).flatten(1).mean(-1)


def hard_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """IoU of the thresholded prediction against the target, per sample (N,).

    Threshold is on the *logit*, and 0.0 is SAM's native mask boundary
    (`mask_threshold` in the predictor), so this number is directly comparable to
    what the GUI shows.
    """
    pred = logits > threshold
    gt = target > 0.5
    inter = (pred & gt).flatten(1).sum(-1).float()
    union = (pred | gt).flatten(1).sum(-1).float()
    # A sample with an empty target and an empty prediction is a perfect match.
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter))


@dataclass
class LossOutput:
    total: torch.Tensor
    dice: torch.Tensor
    bce: torch.Tensor
    iou_head: torch.Tensor
    # Per-sample IoU of the supervised candidate, and which candidate that was.
    iou: torch.Tensor
    best_index: torch.Tensor


class MaskLoss:
    """Weighted Dice + BCE + IoU-head loss over the best-matching candidate."""

    def __init__(self, cfg: LossConfig):
        self.cfg = cfg

    def __call__(
        self,
        logits: torch.Tensor,
        iou_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> LossOutput:
        """`logits` (B, C, H, W), `iou_pred` (B, C), `target` (B, H, W) in {0,1}."""
        batch, candidates = logits.shape[:2]
        flat_logits = logits.flatten(0, 1)                                   # (B*C, H, W)
        flat_target = target.unsqueeze(1).expand(-1, candidates, -1, -1).flatten(0, 1)

        dice = dice_loss(flat_logits, flat_target).view(batch, candidates)
        bce = bce_loss(flat_logits, flat_target).view(batch, candidates)
        with torch.no_grad():
            ious = hard_iou(flat_logits, flat_target).view(batch, candidates)

        # Pick the candidate that actually matches the plume, and supervise it.
        # `ious` is detached, so this is a selection, not a differentiable argmax.
        best = ious.argmax(dim=1)
        rows = torch.arange(batch, device=logits.device)
        dice_best = dice[rows, best]
        bce_best = bce[rows, best]

        # The IoU head is supervised on *every* candidate: at inference the poc
        # ranks candidates by this score, so all three heads need calibration,
        # not just the winner.
        iou_head = F.mse_loss(iou_pred.float(), ious, reduction="none").mean(dim=1)

        total = (
            self.cfg.dice_weight * dice_best
            + self.cfg.bce_weight * bce_best
            + self.cfg.iou_head_weight * iou_head
        )
        return LossOutput(
            total=total.mean(),
            dice=dice_best.mean().detach(),
            bce=bce_best.mean().detach(),
            iou_head=iou_head.mean().detach(),
            iou=ious[rows, best].detach(),
            best_index=best.detach(),
        )
