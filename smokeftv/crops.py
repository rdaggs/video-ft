"""Padded-crop geometry around a prompt box.

This is a deliberate mirror of `smokeseg.geometry.compute_crop` in the poc. PVS
inference is windowed to a padded crop, not run on the full frame — that removes
distant clouds from the field of view and raises effective resolution on small
smoke. Training has to window identically or a train-time IoU means nothing next
to a GUI-time IoU. Keep the two in sync.
"""

from __future__ import annotations

import math

BoxLTRB = tuple[float, float, float, float]


def compute_crop(
    bbox_ltrb: BoxLTRB,
    image_hw: tuple[int, int],
    pad_frac: float = 0.5,
    pad_top_extra: float = 0.5,
) -> tuple[int, int, int, int]:
    """Padded crop (LTRB, pixels) around a bbox.

    Padding is a multiple of the box dimension added per side, applied
    **asymmetrically**: smoke rises and the prompt box sits at the plume *base*,
    so the top gets ``(1 + pad_top_extra)`` x the vertical padding while the
    bottom gets the plain amount. Clamped to the image.

    ``pad_frac <= 0`` returns the full frame (cropping disabled).
    """
    h, w = image_hw
    if pad_frac <= 0:
        return (0, 0, int(w), int(h))
    left, top, right, bottom = (float(v) for v in bbox_ltrb)
    box_w = max(right - left, 1.0)
    box_h = max(bottom - top, 1.0)
    pad_x = pad_frac * box_w
    pad_down = pad_frac * box_h
    pad_up = pad_frac * box_h * (1.0 + max(pad_top_extra, 0.0))
    crop_l = int(math.floor(left - pad_x))
    crop_r = int(math.ceil(right + pad_x))
    crop_t = int(math.floor(top - pad_up))
    crop_b = int(math.ceil(bottom + pad_down))
    crop_l = max(0, min(crop_l, int(w) - 1))
    crop_t = max(0, min(crop_t, int(h) - 1))
    crop_r = max(crop_l + 1, min(crop_r, int(w)))
    crop_b = max(crop_t + 1, min(crop_b, int(h)))
    return (crop_l, crop_t, crop_r, crop_b)


def compute_window(
    bbox_ltrb: BoxLTRB,
    image_hw: tuple[int, int],
    *,
    pad: float = 0.5,
    square: bool = False,
) -> tuple[int, int, int, int]:
    """Uniform-margin window around a bbox, optionally squared. LTRB pixels.

    The margin is a fraction of the box's **longer** side, added equally on all
    four sides, so ``pad=0.5`` makes the window 1.5x the box along that side.
    Deliberately not per-dimension like `compute_crop`: an equal absolute margin
    grows the short axis proportionally more than the long one, which is what
    pulls a lopsided box back toward the square the encoder resizes into. A 16:9
    box therefore gains proportionally more height than width, and a 9:16 box
    more width than height.

    ``square=True`` then extends the short axis out to match the long one, so the
    window is ``(1 + pad)`` x the box's longer side on *both* axes. That is the
    point of the flag: `dataset.py` resizes the crop to a square without
    preserving aspect, so anything not already square is handed to the encoder
    distorted, and a tall plume in a wide crop gets squashed.

    ``pad <= 0`` is legal and means "no margin" — with ``square=True`` that is
    the smallest square containing the box.
    """
    img_h, img_w = image_hw
    left, top, right, bottom = (float(v) for v in bbox_ltrb)
    box_w = max(right - left, 1.0)
    box_h = max(bottom - top, 1.0)
    margin = max(pad, 0.0) / 2.0 * max(box_w, box_h)

    if square:
        # min() with both image dims, not just the matching one: a square window
        # is only square if it fits in *both*, and a frame narrower than the
        # window would otherwise silently return a rectangle.
        side = min(int(round(max(box_w, box_h) + 2 * margin)), int(img_w), int(img_h))
        win_w = win_h = max(side, 1)
    else:
        win_w = max(min(int(round(box_w + 2 * margin)), int(img_w)), 1)
        win_h = max(min(int(round(box_h + 2 * margin)), int(img_h)), 1)

    # Centred on the box, then slid inward to fit rather than clipped. Clipping a
    # square window at a frame edge would make it non-square again — exactly the
    # distortion `square` exists to remove — so the window keeps its shape and
    # moves instead. Only a window larger than the frame loses size, and it was
    # already capped to the frame above.
    left_px = int(round((left + right) / 2.0 - win_w / 2.0))
    top_px = int(round((top + bottom) / 2.0 - win_h / 2.0))
    left_px = max(0, min(left_px, int(img_w) - win_w))
    top_px = max(0, min(top_px, int(img_h) - win_h))
    return (left_px, top_px, left_px + win_w, top_px + win_h)


def pad_box(
    bbox_ltrb: BoxLTRB,
    image_hw: tuple[int, int],
    pad: float = 0.5,
) -> tuple[float, float, float, float]:
    """Grow a box by `pad` x its longer side, clamped to the frame. LTRB floats.

    Used for one thing: turning a corrected polygon's tight bbox into something
    shaped like a *prompt*, for the frames where no Labelbox box exists at all
    (`data.bbox_from_polygon`). A tight bbox around the answer is not a prompt any
    detector produces — the padding is what stops it being one.

    Same margin convention as `compute_window`'s `pad`: half the amount on each
    side, sized off the longer side, so `pad=0.5` grows the box to 1.5x that side.
    Deliberately identical so `data.bbox_from_polygon_pad: 0.5` and
    `data.bbox_pad: 0.5` mean the same shape of thing rather than two conventions
    a reader has to hold apart.

    Floats, not the ints `compute_window` returns: this is a prompt, and every
    other prompt in this codebase is full-frame float pixels straight out of
    `annotations.json`.
    """
    img_h, img_w = image_hw
    left, top, right, bottom = (float(v) for v in bbox_ltrb)
    margin = max(pad, 0.0) / 2.0 * max(right - left, bottom - top, 1.0)
    return (
        max(0.0, left - margin),
        max(0.0, top - margin),
        min(float(img_w), right + margin),
        min(float(img_h), bottom + margin),
    )


def crop_for_box(
    bbox_ltrb: BoxLTRB,
    image_hw: tuple[int, int],
    *,
    pad_frac: float = 0.5,
    pad_top_extra: float = 0.5,
    bbox_pad: float | None = None,
    square_infer: bool = False,
) -> tuple[int, int, int, int]:
    """The one place that decides which windowing a config asks for.

    `bbox_pad: null` keeps the legacy `compute_crop` — per-dimension padding with
    the extra headroom above the box, because smoke rises and the Labelbox box
    sits at the plume base. Setting `bbox_pad` switches to `compute_window` and
    that asymmetry is gone: a centred window cannot also be biased upward. Worth
    knowing before reading a result, since the old geometry is what every number
    recorded so far was measured under.

    Routed through one function so `_drop_tiny` and `__getitem__` cannot disagree
    about the window a sample has — they did not, but nothing stopped them.
    """
    if bbox_pad is None:
        return compute_crop(bbox_ltrb, image_hw, pad_frac, pad_top_extra)
    return compute_window(bbox_ltrb, image_hw, pad=bbox_pad, square=square_infer)


def box_into_crop(
    bbox_ltrb: BoxLTRB, crop: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
    """Map a full-frame box into crop-local pixel coords, clamped to the crop."""
    crop_l, crop_t, crop_r, crop_b = crop
    crop_w, crop_h = crop_r - crop_l, crop_b - crop_t
    left, top, right, bottom = bbox_ltrb
    return (
        min(max(left - crop_l, 0.0), crop_w),
        min(max(top - crop_t, 0.0), crop_h),
        min(max(right - crop_l, 0.0), crop_w),
        min(max(bottom - crop_t, 0.0), crop_h),
    )


# --------------------------------------------------------------------------- #
# Video: one window per track, fixed across every frame of a clip.
# --------------------------------------------------------------------------- #

def track_window(
    boxes,
    image_hw: tuple[int, int],
    *,
    union_pad: float = 0.35,
    square: bool = True,
) -> tuple[int, int, int, int]:
    """One window for a whole track: `compute_window` over the union of its boxes.

    Defined in terms of `compute_window` rather than beside it, so all three crop
    modes share the centre-then-slide-inward rule. That rule is what makes this
    mode work at all: a window clipped at a frame edge stops being square, and
    therefore stops being the same coordinate frame on every frame of the clip —
    which is the entire reason a fixed window exists.

    **Why a fixed window.** In stock SAM 2/3 video the whole frame is encoded, so
    memory features and current-frame features share one coordinate system and
    memory cross-attention can attend positionally. A window that moves frame to
    frame puts the memory bank in a different coordinate frame than the current
    features, and memory attention has to undo an arbitrary similarity transform
    it was never trained for.

    **What it costs.** Measured over the 145 train incidents at 3328x1872: the
    per-masklet union window is 933px at the median against image mode's 384px
    per-frame window, and 30 of 145 saturate the frame's short side. Both resize
    to `image_size`, so that is roughly half the linear resolution on the plume.
    The registration is worth it; the price belongs in any report.
    """
    boxes = list(boxes)
    if not boxes:
        img_h, img_w = image_hw
        return (0, 0, int(img_w), int(img_h))
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    bottom = max(b[3] for b in boxes)
    return compute_window((left, top, right, bottom), image_hw,
                          pad=union_pad, square=square)


def windows_for_clip(crop_cfg, boxes_per_frame, image_hw) -> list[tuple[int, int, int, int]]:
    """The one place that decides which windowing a clip gets.

    Always returns `len(boxes_per_frame)` windows, so `dataset.py` never branches
    on `crop.mode` — the same reason `crop_for_box` exists for the image path.

      full_frame     the frame, repeated
      track_window   one window from every box in the clip, repeated
      per_frame_box  `crop_for_box` per frame (the image repo's geometry)

    Under `track_window` the window is a function of the clip's boxes and nothing
    else — in particular **not** of the prompt-dropout draw. If a dropped frame
    also lost its crop, the model could identify dropped frames from geometry
    alone and the dropout would teach it nothing.
    """
    img_h, img_w = image_hw
    n = len(boxes_per_frame)
    mode = crop_cfg.mode
    if mode == "full_frame":
        return [(0, 0, int(img_w), int(img_h))] * n
    if mode == "track_window":
        flat = [b for boxes in boxes_per_frame for b in (boxes or ())]
        return [track_window(flat, image_hw, union_pad=crop_cfg.union_pad,
                             square=crop_cfg.square_infer)] * n
    if mode == "per_frame_box":
        out = []
        for boxes in boxes_per_frame:
            if not boxes:
                out.append((0, 0, int(img_w), int(img_h)))
                continue
            out.append(crop_for_box(
                boxes[0], image_hw,
                pad_frac=crop_cfg.crop_pad_frac,
                pad_top_extra=crop_cfg.crop_pad_top_extra,
                bbox_pad=crop_cfg.bbox_pad,
                square_infer=crop_cfg.square_infer))
        return out
    raise ValueError(f"crop.mode must be one of full_frame|track_window|"
                     f"per_frame_box; got {mode!r}")
