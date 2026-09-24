"""logits -> the polygon the GUI actually stores, and the two metric surfaces.

**This chain is not in `sam3-finetuned-cvf`.** It lives in the poc, at
`smoke-segmentation-poc/poc/smokeseg/geometry.py`, and the image repo loads it
at runtime rather than reimplementing it — silently substituting a local copy
would produce a number labelled "GUI IoU" that is not one. This repo vendors a
byte-identical copy at `vendor/smokeseg/geometry.py` so `--check` can hash it
offline, and `--check` fails if the vendored copy and the live one diverge.

`iou_polygon` is a *rename* of `polygon_surface(...)["iou"]` — in the image repo
that rename happens in `eval_pipeline/score.py`, and there is no function by
that name anywhere. It is given one here so the metric name in `metrics.jsonl`
and the function that computes it agree.

Two knobs in `PocParams` are dead and are kept anyway, because removing them
would change numbers:
  * `logit_threshold` is never honoured — `fuse_soft` and `hard_iou` both
    binarise at logit 0.0.
  * `max_fill_frac` reaches only the polygon surface, because `metrics.all_ious`
    calls `fuse_soft` without it. See divergence `fuse_fill_frac`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from smokeftv.metrics import all_ious, fuse_soft

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDORED_GEOMETRY = REPO_ROOT / "vendor" / "smokeseg" / "geometry.py"
DEFAULT_POC_ROOT = Path("/home/rileydaggs/projects/smoke-segmentation-poc/poc")


@dataclass(frozen=True)
class PocParams:
    """The post-process chain, pinned.

    Deliberately literals rather than reads of `smokeseg.config.CONFIG`, which
    honours `SMOKESEG_*` environment variables: an eval whose metric definition
    depends on the ambient environment is not reproducible. `--check` compares
    every field against the snapshot's `postprocess:` section, both directions.
    """

    # --- candidate fusion (top-K SAM masks -> one) --- #
    top_k: int = 3
    max_fill_frac: float = 0.9
    logit_threshold: float = 0.0

    # --- clean_mask morphology --- #
    close_kernel: int = 3
    fill_holes: bool = False
    erode_px: int = 0
    keep_largest: bool = True
    min_component_frac: float = 0.0
    smooth_sigma: float = 2.0

    # --- mask_to_polygons --- #
    simplify_eps: float = 1.5
    min_polygon_points: int = 3
    min_polygon_area: float = 8.0
    polygon_smooth_iters: int = 2


def with_overrides(params: PocParams, overrides: dict) -> PocParams:
    known = set(PocParams.__dataclass_fields__)
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"unknown PocParams overrides {unknown}; "
                         f"known are {sorted(known)}")
    return replace(params, **overrides)


def module_fingerprint(module: ModuleType) -> dict:
    """path + sha256[:16] + byte count, so a rerun in a month can only be
    called the same experiment if it matches."""
    path = Path(module.__file__)
    data = path.read_bytes()
    return {"path": str(path),
            "sha256": hashlib.sha256(data).hexdigest()[:16],
            "bytes": len(data)}


def vendored_geometry() -> ModuleType:
    """The in-repo copy. Used by everything; `--check` proves it matches the poc."""
    spec = importlib.util.spec_from_file_location(
        "smokeftv_vendored_geometry", VENDORED_GEOMETRY)
    module = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    spec.loader.exec_module(module)                         # type: ignore[union-attr]
    return module


def load_poc_geometry(poc_root: str | Path = DEFAULT_POC_ROOT) -> ModuleType:
    """The LIVE poc module. Only `--check` should need this."""
    root = str(Path(poc_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from smokeseg import geometry                            # noqa: PLC0415
    return geometry


def upsample_logits(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinear-upsample decoder logits to `size` — one implementation, one place.

    The same bilinear step `SAM2Transforms.postprocess_masks` takes; its
    hole/sprinkle cleanup is inference-only and skipped. Named rather than
    inlined because the two surfaces need the *same* forward at two different
    resolutions — loss space for `iou_fused`, crop space for the polygon — and a
    second copy would let them drift while still being reported as measurements
    of one prediction.
    """
    if tuple(logits.shape[-2:]) == tuple(size):
        return logits
    return F.interpolate(logits.float(), tuple(size), mode="bilinear",
                         align_corners=False)


def predicted_polygons(
    native_logits_one: torch.Tensor,
    iou_pred_one: torch.Tensor,
    crop: tuple[int, int, int, int],
    params: PocParams,
    geometry: ModuleType | None = None,
) -> list[np.ndarray]:
    """The polygons this prediction would store, in **crop-local** pixels.

    Fuse at crop resolution -> `clean_mask` -> `mask_to_polygons`: the poc's own
    chain, and the only thing that reaches `polygons.coco.json`. Each polygon is
    a (K, 2) int32 array of (x, y) offsets from the crop's top-left corner.

    Split out of `polygon_surface` so anything *drawing* the prediction draws the
    same vertices `iou_polygon` was computed from.

    `native_logits_one` is (C, 288, 288), `iou_pred_one` is (C,).
    """
    geometry = geometry or vendored_geometry()
    crop_l, crop_t, crop_r, crop_b = (int(v) for v in crop)
    crop_h, crop_w = max(crop_b - crop_t, 1), max(crop_r - crop_l, 1)

    # Crop-native resolution, NOT loss_size: max_fill_frac is then measured
    # against real crop area and the morphology kernels are in frame pixels.
    logits = upsample_logits(native_logits_one[None], (crop_h, crop_w))
    fused = fuse_soft(logits, iou_pred_one[None], top_k=params.top_k,
                      max_fill_frac=params.max_fill_frac)[0]
    fused_np = fused.detach().cpu().numpy()

    # Morphology runs on the UNPADDED crop — the crop's own edge is where the
    # poc's kernels see a border too.
    cleaned = geometry.clean_mask(
        fused_np,
        close_kernel=params.close_kernel,
        fill_holes=params.fill_holes,
        erode_px=params.erode_px,
        keep_largest=params.keep_largest,
        min_component_frac=params.min_component_frac,
        smooth_sigma=params.smooth_sigma,
    )
    # One pixel of zero padding so a plume touching the crop edge traces a
    # CLOSED contour, exactly as it does after the poc pastes the crop back.
    padded = np.pad(cleaned.astype(np.uint8), 1)
    polys = geometry.mask_to_polygons(
        padded,
        simplify_eps=params.simplify_eps,
        min_points=params.min_polygon_points,
        min_area=params.min_polygon_area,
        smooth_iters=params.polygon_smooth_iters,
    )
    # Undo the pad, so the vertices are in the crop's own coordinate frame.
    # Losing this line moves every vertex by one pixel and nothing crashes.
    return [np.asarray(p, dtype=np.int32) - 1 for p in polys]


def _counts(pred_px: int, gt_px: int, inter_px: int) -> dict[str, float]:
    """IoU / precision / recall from three pixel counts.

    Empty-against-empty is a perfect match, matching `metrics.iou_from_masks`
    and `losses.hard_iou`; an empty prediction against a real plume scores 0,
    not NaN, so the worst-frame table can rank it.
    """
    union = pred_px + gt_px - inter_px
    both_empty = pred_px == 0 and gt_px == 0
    return {
        "iou": 1.0 if both_empty else (inter_px / union if union > 0 else 1.0),
        "precision": 1.0 if pred_px == 0 and both_empty else (
            inter_px / pred_px if pred_px > 0 else 0.0),
        "recall": 1.0 if gt_px == 0 and both_empty else (
            inter_px / gt_px if gt_px > 0 else 0.0),
    }


def polygon_surface(
    native_logits_one: torch.Tensor,
    iou_pred_one: torch.Tensor,
    crop: tuple[int, int, int, int],
    gt_polygons,
    image_hw: tuple[int, int],
    params: PocParams,
    geometry: ModuleType | None = None,
) -> dict[str, float]:
    """Score one sample the way the GUI would see it.

    `predicted_polygons` -> rasterize them back -> compare against the GT
    polygons rasterized at FULL-FRAME resolution. Scoring the *rasterized
    polygons* rather than the cleaned mask is deliberate: Douglas-Peucker and
    Chaikin both move the boundary, and the polygon is the artifact that reaches
    `polygons.coco.json`.

    The asymmetry is the subtlest thing here and it is on purpose: `pred_px` is
    crop-sized, `gt_px` is full-frame. **GT outside the crop window counts
    against recall**, because the product misses it too — the poc's PVS path is
    windowed to the same padded crop. It is reported as `gt_px_outside_crop`, so
    a windowing problem reads as a windowing problem rather than as a decoder
    that lost recall.
    """
    geometry = geometry or vendored_geometry()
    crop_l, crop_t, crop_r, crop_b = (int(v) for v in crop)
    crop_h, crop_w = max(crop_b - crop_t, 1), max(crop_r - crop_l, 1)

    polys = predicted_polygons(native_logits_one, iou_pred_one, crop,
                               params, geometry)

    pred = np.zeros((crop_h, crop_w), dtype=np.uint8)
    if polys:
        cv2.fillPoly(pred, polys, color=1)
    pred_bool = pred.astype(bool)

    height, width = (int(v) for v in image_hw)
    gt_full = np.zeros((height, width), dtype=np.uint8)
    contours = []
    for poly in gt_polygons or ():
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            contours.append(np.round(pts).astype(np.int32))
    if contours:
        cv2.fillPoly(gt_full, contours, color=1)
    gt_bool = gt_full.astype(bool)
    gt_in_crop = gt_bool[crop_t:crop_t + crop_h, crop_l:crop_l + crop_w]

    pred_px = int(pred_bool.sum())
    gt_px = int(gt_bool.sum())
    inter_px = int(np.logical_and(pred_bool, gt_in_crop).sum())
    return {
        **_counts(pred_px, gt_px, inter_px),
        "pred_px": pred_px,
        "gt_px": gt_px,
        "inter_px": inter_px,
        "n_polygons": len(polys),
        "gt_px_outside_crop": gt_px - int(gt_in_crop.sum()),
    }


def iou_polygon(*args, **kwargs) -> float:
    """`polygon_surface(...)["iou"]`, named so the metric and the function agree."""
    return float(polygon_surface(*args, **kwargs)["iou"])


def fused_surface(
    native_logits: torch.Tensor,
    iou_pred: torch.Tensor,
    target: torch.Tensor,
    loss_size: int,
    params: PocParams,
) -> dict[str, torch.Tensor]:
    """Per-sample scores for the raw fused mask, in loss space.

    The three IoUs come from `metrics.all_ious` **unchanged** — that is the
    whole point of this surface. `iou_fused` from here and `iou_fused` in the
    training log are then the same number by construction.
    """
    logits = upsample_logits(native_logits, (loss_size, loss_size))
    ious = all_ious(logits, iou_pred, target, top_k=params.top_k)
    # A second fuse, only for the pixel counts. Deliberately not folded into
    # `all_ious`: keeping that call verbatim is what keeps this surface
    # comparable to the training log. See divergence `fuse_fill_frac`.
    fused = fuse_soft(logits, iou_pred, top_k=params.top_k,
                      max_fill_frac=params.max_fill_frac)
    gt = target > 0.5
    return {
        **ious,
        "pred_px": fused.flatten(1).sum(-1),
        "gt_px": gt.flatten(1).sum(-1),
        "inter_px": (fused & gt).flatten(1).sum(-1),
    }
