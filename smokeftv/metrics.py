"""IoU, measured three ways, and a running accumulator.

The number the training loss drives is the *oracle* IoU — the best of the three
decoder candidates. Inference cannot see which one that is. So reporting only the
oracle would flatter the model and hide a regression in candidate ranking. Three
numbers are tracked instead:

  ``iou_best``      best candidate. Oracle ceiling; the training objective.
  ``iou_selected``  the candidate the IoU head scores highest. What you get from
                    the poc with ``mask_combine_top_k = 1``.
  ``iou_fused``     mean of the top-k candidates' probability maps, thresholded.
                    What the poc actually does (``mask_combine_mode = "soft"``,
                    ``top_k = 3``).

``iou_fused`` is the one to quote. It is not a perfect stand-in for a GUI IoU —
`clean_mask` morphology, smoothing and polygonization all still happen downstream
in the poc — but it is measuring the same object the GUI shows.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from smokeftv.losses import hard_iou


def fuse_soft(
    logits: torch.Tensor,
    scores: torch.Tensor,
    top_k: int = 3,
    max_fill_frac: float = 0.9,
) -> torch.Tensor:
    """Fuse candidates into one mask per sample, as `fuse_candidate_masks` does.

    `logits` is (B, C, H, W), `scores` is (B, C). Candidates filling more than
    `max_fill_frac` of the frame are dropped as degenerate before ranking; if that
    leaves nothing, the smallest non-empty candidate is kept (the poc's
    accept_degenerate fallback, which the box-came-from-a-detector case relies on).
    Returns a bool mask (B, H, W).
    """
    batch, candidates, height, width = logits.shape
    probs = logits.float().sigmoid()
    binary = probs > 0.5
    area = binary.flatten(2).sum(-1).float()                       # (B, C)
    fill = area / float(height * width)

    usable = (area > 0) & (fill <= max_fill_frac)
    # Rank by predicted quality, pushing unusable candidates to the back.
    order = torch.argsort(
        torch.where(usable, scores.float(), scores.float() - 1e4), dim=1, descending=True
    )

    out = torch.zeros((batch, height, width), dtype=torch.bool, device=logits.device)
    for i in range(batch):
        keep = [int(c) for c in order[i] if usable[i, int(c)]][:top_k]
        if not keep:
            nonempty = [int(c) for c in range(candidates) if area[i, c] > 0]
            if not nonempty:
                continue
            keep = [min(nonempty, key=lambda c: float(area[i, c]))]
        out[i] = probs[i, keep].mean(dim=0) > 0.5
    return out


def iou_from_masks(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample IoU between two bool/{0,1} masks of shape (B, H, W)."""
    p, g = pred > 0.5, target > 0.5
    inter = (p & g).flatten(1).sum(-1).float()
    union = (p | g).flatten(1).sum(-1).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter))


def all_ious(
    logits: torch.Tensor,
    iou_pred: torch.Tensor,
    target: torch.Tensor,
    top_k: int = 3,
) -> dict[str, torch.Tensor]:
    """The three per-sample IoUs described in the module docstring."""
    batch, candidates = logits.shape[:2]
    rows = torch.arange(batch, device=logits.device)

    per_candidate = hard_iou(
        logits.flatten(0, 1),
        target.unsqueeze(1).expand(-1, candidates, -1, -1).flatten(0, 1),
    ).view(batch, candidates)

    return {
        "iou_best": per_candidate.max(dim=1).values,
        "iou_selected": per_candidate[rows, iou_pred.float().argmax(dim=1)],
        "iou_fused": iou_from_masks(fuse_soft(logits, iou_pred, top_k=top_k), target),
    }


class Meter:
    """Running means, plus per-incident means for the eval report."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.by_incident: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.incident_counts: dict[str, int] = defaultdict(int)

    def add(self, values: dict[str, torch.Tensor | float], incidents: list[str] | None = None):
        for name, value in values.items():
            if torch.is_tensor(value) and value.numel() > 1:
                self.sums[name] += float(value.sum())
                self.counts[name] += value.numel()
                if incidents is not None:
                    for incident, item in zip(incidents, value.tolist()):
                        self.by_incident[incident][name] += item
            else:
                self.sums[name] += float(value)
                self.counts[name] += 1
        if incidents is not None:
            for incident in incidents:
                self.incident_counts[incident] += 1

    def mean(self, name: str) -> float:
        return self.sums[name] / self.counts[name] if self.counts[name] else float("nan")

    def means(self) -> dict[str, float]:
        return {name: self.mean(name) for name in self.sums}

    def incident_means(self) -> dict[str, dict[str, float]]:
        return {
            incident: {
                name: total / self.incident_counts[incident]
                for name, total in values.items()
            }
            for incident, values in self.by_incident.items()
        }
