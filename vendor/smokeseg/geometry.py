"""Mask <-> polygon <-> RLE conversions.

Kept separate from rendering so the data layer can convert masks to compact,
serializable forms (polygons / COCO RLE) without importing matplotlib or Gradio.
"""

from __future__ import annotations

import cv2
import numpy as np

try:  # pycocotools is optional; RLE just degrades gracefully without it.
    from pycocotools import mask as coco_mask  # type: ignore

    _HAS_COCO = True
except Exception:  # pragma: no cover
    _HAS_COCO = False


BoxLTRB = tuple[int, int, int, int]


def compute_crop(
    bbox_ltrb: BoxLTRB,
    image_hw: tuple[int, int],
    pad_frac: float = 2.0,
    pad_top_extra: float = 0.5,
) -> BoxLTRB:
    """Padded crop (LTRB, pixels) around a bbox for windowed inference (R2).

    Padding is a multiple of the box dimension added per side, applied
    **asymmetrically** — smoke rises and the detector box sits at the plume
    *base*, so the top gets ``(1 + pad_top_extra)`` × the vertical padding while
    the bottom gets the plain amount. The result is clamped to the image.

    ``pad_frac <= 0`` returns the full frame (cropping disabled).
    """
    h, w = image_hw
    if pad_frac <= 0:
        return (0, 0, int(w), int(h))
    l, t, r, b = (float(v) for v in bbox_ltrb)
    bw = max(r - l, 1.0)
    bh = max(b - t, 1.0)
    pad_x = pad_frac * bw
    pad_down = pad_frac * bh
    pad_up = pad_frac * bh * (1.0 + max(pad_top_extra, 0.0))
    cl = int(np.floor(l - pad_x))
    cr = int(np.ceil(r + pad_x))
    ct = int(np.floor(t - pad_up))
    cb = int(np.ceil(b + pad_down))
    cl = max(0, min(cl, int(w) - 1))
    ct = max(0, min(ct, int(h) - 1))
    cr = max(cl + 1, min(cr, int(w)))
    cb = max(ct + 1, min(cb, int(h)))
    return (cl, ct, cr, cb)


def close_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological close (dilate then erode) with an elliptical kernel of the
    given pixel ``radius`` — bridges gaps up to ~2·radius between fragments (R5).
    ``radius <= 0`` is a no-op."""
    m = (mask > 0).astype(np.uint8)
    if radius <= 0:
        return m.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(bool)


def chaikin_smooth(pts: np.ndarray, iters: int) -> np.ndarray:
    """Round a closed polygon into a smooth curve via Chaikin corner-cutting.

    Each pass replaces every vertex with two points at 1/4 and 3/4 along its two
    incident edges, shaving sharp corners. Two passes turn a faceted polygon into
    a smoothly curving outline — the difference between a jagged Douglas–Peucker
    trace and a soft, smoke-like boundary. Point count doubles per pass, so feed
    it an already-simplified polygon and keep ``iters`` small (1–3).
    """
    p = np.asarray(pts, dtype=np.float32)
    for _ in range(max(0, int(iters))):
        if len(p) < 3:
            break
        nxt = np.roll(p, -1, axis=0)
        q = p + 0.25 * (nxt - p)      # point 1/4 along each edge
        r = p + 0.75 * (nxt - p)      # point 3/4 along each edge
        out = np.empty((2 * len(p), 2), dtype=np.float32)
        out[0::2] = q
        out[1::2] = r
        p = out
    return p


def mask_to_polygons(
    mask: np.ndarray,
    simplify_eps: float = 1.5,
    min_points: int = 3,
    min_area: float = 8.0,
    smooth_iters: int = 0,
) -> list[np.ndarray]:
    """Extract polygon(s) from a boolean mask via OpenCV contours.

    Returns a list of (K, 2) int arrays (x, y): Douglas–Peucker simplified, then
    (when ``smooth_iters > 0``) Chaikin-rounded into smooth, smoke-like curves.
    Only external contours are returned (holes are ignored — fine for smoke).
    """
    if mask is None or mask.sum() == 0:
        return []
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[np.ndarray] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        if simplify_eps and simplify_eps > 0:
            cnt = cv2.approxPolyDP(cnt, epsilon=simplify_eps, closed=True)
        pts = cnt.reshape(-1, 2)
        if pts.shape[0] < min_points:
            continue
        if smooth_iters > 0 and pts.shape[0] >= 3:
            pts = chaikin_smooth(pts, smooth_iters)
        polys.append(np.round(pts).astype(np.int32))
    return polys


def polygons_to_mask(polys: list[np.ndarray], image_hw: tuple[int, int]) -> np.ndarray:
    """Rasterize polygons back into a boolean mask of shape image_hw."""
    h, w = image_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if polys:
        cv2.fillPoly(mask, [np.asarray(p, dtype=np.int32) for p in polys], color=1)
    return mask.astype(bool)


def encode_rle(mask: np.ndarray) -> dict:
    """Encode a boolean mask as COCO RLE (counts as utf-8 str for JSON)."""
    m = np.asfortranarray((mask > 0).astype(np.uint8))
    if _HAS_COCO:
        rle = coco_mask.encode(m)
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}
    # Fallback: simple flattened run-length (not COCO-compatible, but round-trips).
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": _rle_fallback(m)}


def decode_rle(rle: dict) -> np.ndarray:
    """Decode COCO RLE (or the fallback format) back to a boolean mask."""
    size = rle["size"]
    counts = rle["counts"]
    if _HAS_COCO and isinstance(counts, str):
        r = {"size": size, "counts": counts.encode("ascii")}
        return coco_mask.decode(r).astype(bool)
    if isinstance(counts, list):
        return _rle_fallback_decode(counts, tuple(size)).astype(bool)
    # COCO available but counts is already bytes
    if _HAS_COCO:
        return coco_mask.decode({"size": size, "counts": counts}).astype(bool)
    raise ValueError("Cannot decode RLE without pycocotools for this format")


# --------------------------------------------------------------------------- #
# Minimal column-major RLE fallback (only used if pycocotools is unavailable).
# --------------------------------------------------------------------------- #
def _rle_fallback(m_fortran: np.ndarray) -> list[int]:
    flat = m_fortran.flatten(order="F")
    counts: list[int] = []
    prev = 0  # COCO RLE starts counting zeros
    run = 0
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            prev = v
            run = 1
    counts.append(run)
    return counts


def _rle_fallback_decode(counts: list[int], size: tuple[int, int]) -> np.ndarray:
    h, w = size
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        if val == 1:
            flat[idx:idx + c] = 1
        idx += c
        val ^= 1
    return flat.reshape((h, w), order="F")

def filter_components(
    mask: np.ndarray,
    keep_largest: bool = False,
    min_component_frac: float = 0.0,
) -> np.ndarray:
    """Drop small disconnected blobs, keeping only substantial component(s).

    This is the cure for "lots of little polygons alongside the main one": the
    binarized mask often carries speckle that ``mask_to_polygons`` would emit as
    its own tiny polygon. We keep either only the single largest connected
    component (``keep_largest``) or every component whose area is at least
    ``min_component_frac`` of the largest. ``keep_largest`` wins when both are
    set. With both off this is a no-op.
    """
    m = (mask > 0).astype(np.uint8)
    if not keep_largest and min_component_frac <= 0.0:
        return m.astype(bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 2:                       # background + (at most) one component
        return m.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]   # label 0 is background
    largest_area = int(areas.max())
    if keep_largest:
        keep = {int(np.argmax(areas)) + 1}
    else:
        thr = max(1.0, float(min_component_frac) * largest_area)
        keep = {i + 1 for i, a in enumerate(areas) if a >= thr}
    return np.isin(labels, list(keep))


def smooth_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Round a binary mask's boundary by Gaussian-blurring it and re-thresholding
    at 0.5 — the simplest way to turn a jagged pixel edge into a soft, organic,
    smoke-like outline. ``sigma <= 0`` is a no-op. Cannot create new components
    (it only reshapes existing boundaries), so run it after ``filter_components``.
    """
    if sigma <= 0:
        return (mask > 0)
    m = (mask > 0).astype(np.float32)
    k = int(2 * round(3 * sigma) + 1)   # odd kernel spanning ~±3σ
    blurred = cv2.GaussianBlur(m, (k, k), float(sigma))
    return blurred >= 0.5


def temporal_blend(
    masks: list[np.ndarray | None],
    weights: list[float],
    thresh: float = 0.5,
) -> np.ndarray | None:
    """Weighted temporal average of same-shape boolean masks, re-thresholded.

    ``masks`` are the union masks of the frames in a time window (aligned, same
    H×W); a ``None`` slot is a window frame with no mask and is skipped, its
    ``weights`` entry dropped from the denominator too. A pixel survives when its
    weighted "on" fraction reaches ``thresh`` (0.5 = majority-by-weight). This is
    the frame-to-frame de-flicker that makes played-back smoke evolve smoothly
    instead of boiling — the jaggedness that differs each frame averages out while
    a shape present across the window is preserved. Returns ``None`` when no slot
    carried a mask.
    """
    acc: np.ndarray | None = None
    wsum = 0.0
    for m, w in zip(masks, weights):
        if m is None or w <= 0:
            continue
        acc = (m > 0).astype(np.float32) * float(w) if acc is None \
            else acc + (m > 0).astype(np.float32) * float(w)
        wsum += float(w)
    if acc is None or wsum <= 0:
        return None
    return (acc / wsum) >= thresh


def clean_mask(
    mask: np.ndarray,
    close_kernel: int = 5,
    fill_holes: bool = True,
    erode_px: int = 0,
    keep_largest: bool = False,
    min_component_frac: float = 0.0,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Bridge small gaps (close) -> optionally fill holes -> drop stray blobs ->
    smooth the boundary -> optionally erode inward.

    For diffuse smoke a large ``close_kernel`` rounds concave corners (turning a
    triangular plume oval) and ``fill_holes`` solidifies translucent gaps, both
    of which loosen the outline. ``keep_largest`` / ``min_component_frac`` remove
    the little stray polygons; ``smooth_sigma`` softens the edge into a smoke-like
    curve; ``erode_px`` is a direct uniform inward pull for a tighter boundary.
    """
    m = (mask > 0).astype(np.uint8)
    if close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if fill_holes:
        # Re-fill each external contour solid â no interior holes.
        filled = np.zeros_like(m)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, cnts, -1, color=1, thickness=cv2.FILLED)
        m = filled
    if keep_largest or min_component_frac > 0.0:
        m = filter_components(m, keep_largest, min_component_frac).astype(np.uint8)
    if smooth_sigma > 0.0:
        m = smooth_mask(m, smooth_sigma).astype(np.uint8)
    if erode_px > 0:
        ek = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        m = cv2.erode(m, ek)
    return m.astype(bool)