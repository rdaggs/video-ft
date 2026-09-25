"""Rasterising GT polygons into the crop's coordinate frame."""

from __future__ import annotations

import cv2
import numpy as np


def _to_crop_space(polygons, crop, out_size: int):
    """Full-frame polygon vertices -> loss-space pixels.

    Scaling is **anisotropic** — each axis independently — because
    `dataset.build_transform` resizes the crop to a square without preserving
    aspect. Vertices are transformed and then filled ONCE; rasterising and then
    resampling would introduce half-pixel drift between the target and the
    prediction it is compared against.
    """
    crop_l, crop_t, crop_r, crop_b = (int(v) for v in crop)
    crop_w, crop_h = max(crop_r - crop_l, 1), max(crop_b - crop_t, 1)
    scale_x, scale_y = out_size / crop_w, out_size / crop_h
    out = []
    for poly in polygons or ():
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            continue
        pts = pts.copy()
        pts[:, 0] = (pts[:, 0] - crop_l) * scale_x
        pts[:, 1] = (pts[:, 1] - crop_t) * scale_y
        out.append(np.round(pts).astype(np.int32))
    return out


def polygons_to_mask(polygons, crop, out_size: int) -> np.ndarray:
    """(out_size, out_size) uint8 target mask, 1 inside the plume."""
    mask = np.zeros((out_size, out_size), dtype=np.uint8)
    contours = _to_crop_space(polygons, crop, out_size)
    if contours:
        cv2.fillPoly(mask, contours, color=1)
    return mask


def polygon_area_in_loss_space(polygons, crop, out_size: int) -> float:
    """Target area without touching an image — used to drop samples early."""
    return float(polygons_to_mask(polygons, crop, out_size).sum())
