"""Box arithmetic, the annotations reader, carry-forward, and the two inert
prompt transforms.

Ported from `sam3-finetuned-cvf/smokeft/bbox_event_grouping.py`,
`bbox_boundary_extend.py` and `bbox_repair.py`. Same standalone contract as all
three: **stdlib only** — no torch, no numpy, nothing from the rest of this
package — so the geometry can be copied into the poc, and so
`config_all --check` and the probes never import CUDA.

What did NOT come across: `bbox_event_grouping`'s union modes. Its *linking*
step is masklet assignment and lives in `smokeftv.masklets`; the union modes are
a prompt transform and a separate experiment (as a blanket transform they buy
2.8 points of coverage and pay 0.10 of box IoU). `EventGroupingParams` and
`BoundaryExtendParams` exist here only so the snapshot's keys survive
unknown-key rejection; both raise when enabled.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

BoxLTRB = tuple[float, float, float, float]

# `bbox_from_polygon`'s marker, carried in a sample's `box_gap`. Negative so
# that gap > 0 ("carried forward") and gap == 0 ("the frame's own box") both
# keep their meaning.
POLYGON_BOX_GAP = -1
# No box at all, and none to carry. Distinct from POLYGON_BOX_GAP so the two
# cannot be confused in a manifest.
NO_BOX_GAP = -2


# --------------------------------------------------------------------------- #
# Box arithmetic. Kept local so this file has no dependencies.
# --------------------------------------------------------------------------- #

def box_area(box: BoxLTRB) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def box_iou(a: BoxLTRB, b: BoxLTRB) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_union(boxes: Sequence[BoxLTRB]) -> BoxLTRB:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def box_gap_frac(a: BoxLTRB, b: BoxLTRB) -> float:
    """Edge-to-edge separation as a fraction of the larger box's longer side.

    0.0 when the boxes touch or overlap. Scale-relative because the corpus spans
    12-pixel bases and 900-pixel plumes, and "80 pixels apart" means opposite
    things at those two sizes. Chebyshev (`max(dx, dy)`), not Euclidean.
    """
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    scale = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1.0)
    return max(dx, dy) / scale


def clamp_box(box: BoxLTRB, image_hw: tuple[int, int] | None) -> BoxLTRB:
    """Clamp to the frame when the frame's size is known. LTRB floats.

    `image_hw` is optional throughout because `annotations.json` does not record
    it — a caller that has the frames (or the COCO file) can supply it, and a
    caller that only has boxes still gets a prompt.
    """
    if image_hw is None:
        return box
    height, width = image_hw
    return (max(0.0, box[0]), max(0.0, box[1]),
            min(float(width), box[2]), min(float(height), box[3]))


def bbox_of_polygons(polygons: Sequence[Sequence[float]]) -> BoxLTRB | None:
    """Tight LTRB around flat `[x0, y0, x1, y1, ...]` rings. None if empty."""
    xs: list[float] = []
    ys: list[float] = []
    for poly in polygons or ():
        xs.extend(float(v) for v in poly[0::2])
        ys.extend(float(v) for v in poly[1::2])
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------------- #
# Reading, and carrying forward.
# --------------------------------------------------------------------------- #

def boxes_from_annotations(path: str | Path) -> dict[int, list[BoxLTRB]]:
    """Labelbox GT boxes per frame index, LTRB pixels. Boxless frames omitted.

    Omitted, not stored as an empty list: leading frames legitimately have no
    box because the plume has not appeared yet, and `resolve_box`'s bisect wants
    a list of frames that actually carry one.

    The masklet linker and the clip builder both read boxes through here rather
    than keeping their own parser — two readers of one JSON file is how the
    linker ends up grouping boxes training never prompted with.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[int, list[BoxLTRB]] = {}
    for frame in data.get("frames", []):
        boxes = []
        for ann in frame.get("annotations") or []:
            box = ann.get("bbox")
            if not box:
                continue
            left, top = float(box["left"]), float(box["top"])
            boxes.append((left, top,
                          left + float(box["width"]), top + float(box["height"])))
        if boxes:
            out[int(frame["frame_index"])] = boxes
    return out


def resolve_box(
    frame_index: int,
    boxes: Mapping[int, list[BoxLTRB]],
    propagate: int = -1,
    box_frames: list[int] | None = None,
) -> tuple[list[BoxLTRB], int] | None:
    """The prompt box(es) for a frame, carried forward from the nearest earlier one.

    Labelbox stops boxing a plume well before the labeler stops correcting masks:
    an incident whose annotations end at frame 33 can carry polygons to frame 45,
    and those 12 frames are real corrected masks drawn under the frame-33 box,
    which is what the GUI was still showing. `propagate < 0` carries a box forward
    however far it has to go; a `propagate >= 0` cap exists for ablating that, and
    `0` means own-box-only.

    Never looks ahead: a later box is not what the labeler saw. Returns
    `(boxes, gap)` where gap is how many frames back the box came from, or None
    when there is nothing to carry.
    """
    frames = sorted(boxes) if box_frames is None else box_frames
    position = bisect_right(frames, frame_index)
    if position == 0:
        return None
    source = frames[position - 1]
    gap = frame_index - source
    if propagate >= 0 and gap > propagate:
        return None
    return list(boxes[source]), gap


# --------------------------------------------------------------------------- #
# Param coercion, shared by the three blocks below.
# --------------------------------------------------------------------------- #

def _coerce(cls, value: Any, dotted: str, nested: Mapping[str, Any] | None = None):
    """Build `cls` from a mapping, rejecting unknown keys.

    Unknown keys raise for the same reason `config._from_dict` raises: a typo'd
    key that silently does nothing is a wasted GPU-hour and a confusing plot.
    """
    if isinstance(value, cls):
        value.validate()
        return value
    if value is None:
        out = cls()
        out.validate()
        return out
    if not isinstance(value, Mapping):
        raise ValueError(f"{dotted} must be a mapping; got {value!r}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(
            f"{dotted}: unknown keys {unknown}. Known keys are {sorted(known)}.")
    kwargs = dict(value)
    for name, sub in (nested or {}).items():
        if name in kwargs:
            kwargs[name] = sub.coerce(kwargs[name], f"{dotted}.{name}")
    out = cls(**kwargs)
    out.validate()
    return out


def _positive(name: str, value: Any, low: float = 0.0,
              high: float | None = None, dotted: str = "") -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{dotted}.{name} must be a number; got {value!r}")
    if float(value) < low or (high is not None and float(value) > high):
        bound = f"[{low}, {high}]" if high is not None else f">= {low}"
        raise ValueError(f"{dotted}.{name} must be {bound}; got {value!r}")


# --------------------------------------------------------------------------- #
# bbox_repair — the only one of the three that is live, and only for eval arm 1.
# --------------------------------------------------------------------------- #

@dataclass
class BboxRepairParams:
    """`prompting.bbox_repair`. Off by default.

    Repairs a prompt box that fails to contain its own attributed GT, from the
    GT already paired with it. Never touches labels on disk, and is a bit-exact
    no-op on a clean frame.

    **Off for training**, because it reads the GT to rewrite the prompt: a model
    trained under it learns a prompt no detector produces. On for eval arm 1,
    which is the arm that has to reproduce `eval_video.baseline_iou_polygon` —
    that number was measured with it enabled, and without it the same weights
    score 0.6577 instead of 0.6621.
    """

    enabled: bool = False
    # A box already within this many pixels of containing its GT on every side
    # is left untouched — the no-op that keeps a clean frame bit-exact.
    tol_px: float = 2.0
    # Growth applied after unioning the box with the GT bbox, as a fraction of
    # the unioned box per axis split evenly across the two sides: 0.5 => +25%
    # each side. Minimal by design — crop padding already supplies the context
    # the encoder was tuned for, so the repair only has to make the GT
    # reachable, not re-pad it.
    pad_frac: float = 0.5
    # A repaired box may not exceed this fraction of the frame area. A box that
    # would is a label so broken that expanding it hides the error; it is left
    # unrepaired and flagged for review instead.
    max_area_frac: float = 0.25

    def validate(self) -> None:
        d = "prompting.bbox_repair"
        if not isinstance(self.enabled, bool):
            raise ValueError(f"{d}.enabled must be true or false; got {self.enabled!r}")
        _positive("tol_px", self.tol_px, 0.0, dotted=d)
        _positive("pad_frac", self.pad_frac, 0.0, dotted=d)
        _positive("max_area_frac", self.max_area_frac, 0.0, 1.0, dotted=d)

    @classmethod
    def coerce(cls, value: Any, dotted: str = "prompting.bbox_repair"):
        return _coerce(cls, value, dotted)


def repair_box(
    prompt_box: BoxLTRB,
    gt_box: BoxLTRB | None,
    image_hw: tuple[int, int],
    params: BboxRepairParams | None = None,
) -> tuple[BoxLTRB, str]:
    """Grow `prompt_box` to contain `gt_box`, or leave it untouched.

    Returns `(box, reason)` where reason is one of:
      * ``"no-gt"``   — no GT bbox to repair against; box unchanged.
      * ``"clean"``   — already contains GT within `tol_px`; **bit-exact** unchanged.
      * ``"repaired"`` — grown to contain GT (unioned, padded, clipped).
      * ``"rejected-too-large"`` — repair would exceed `max_area_frac`; unchanged.

    Deterministic and idempotent: a repaired box already contains its GT, so a
    second call returns "clean" with the same coordinates.

    **`params.enabled` is deliberately NOT read here.** The reference behaves the
    same way — the caller decides whether to call at all — and keeping the gate
    at the call site is what lets eval arm 1 repair while training does not,
    from one code path whose arithmetic cannot drift between the two.
    """
    params = BboxRepairParams.coerce(params)
    prompt: BoxLTRB = tuple(float(v) for v in prompt_box)  # type: ignore[assignment]
    if gt_box is None:
        return prompt, "no-gt"
    if _contains(prompt, gt_box, params.tol_px):
        return prompt, "clean"

    box = _union(prompt, gt_box)
    # Pad per axis: pad_frac split across the two sides (0.5 => +25% each side).
    half = params.pad_frac / 2.0
    box_w = max(box[2] - box[0], 1.0)
    box_h = max(box[3] - box[1], 1.0)
    box = (box[0] - half * box_w, box[1] - half * box_h,
           box[2] + half * box_w, box[3] + half * box_h)
    box = _clip(box, image_hw)

    frame_area = float(image_hw[0]) * float(image_hw[1])
    if frame_area > 0 and box_area(box) / frame_area > params.max_area_frac:
        return prompt, "rejected-too-large"
    return box, "repaired"


def _contains(outer: BoxLTRB, inner: BoxLTRB, tol: float) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def _union(a: BoxLTRB, b: BoxLTRB) -> BoxLTRB:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _clip(box: BoxLTRB, image_hw: tuple[int, int]) -> BoxLTRB:
    h, w = float(image_hw[0]), float(image_hw[1])
    return (max(0.0, box[0]), max(0.0, box[1]), min(w, box[2]), min(h, box[3]))


# --------------------------------------------------------------------------- #
# The two inert blocks. Present so the snapshot's keys survive unknown-key
# rejection; both refuse to be enabled.
# --------------------------------------------------------------------------- #

@dataclass
class EventGroupingParams:
    """`prompting.bbox_event_grouping`. OFF, and it raises if turned on.

    Its *linking* step is masklet assignment and lives in `smokeftv.masklets`,
    which is where `link_iou`, `link_gap_frac` and `max_frame_gap` are actually
    read (under `masklets:`). What is missing here are the union modes, which
    are a prompt transform and a different experiment.
    """

    enabled: bool = False
    mode: str = "window_union"
    link_iou: float = 0.1
    link_gap_frac: float = 0.5
    max_frame_gap: int = 5
    window: int = 5
    causal: bool = True
    ema_alpha: float = 0.5
    outlier_area_ratio: float = 8.0
    max_area_growth: float = 8.0
    fill_span: bool = False

    def validate(self) -> None:
        if self.enabled:
            raise NotImplementedError(
                "prompting.bbox_event_grouping.enabled: true — the union modes "
                "are not ported. Its LINKING step is masklet assignment and "
                "lives in smokeftv.masklets (see the `masklets:` section, which "
                "is where link_iou / link_gap_frac / max_frame_gap are read). "
                "The prompt transform is a separate experiment — as a blanket "
                "transform it buys 2.8 points of coverage and pays 0.10 of box "
                "IoU — and turning it on makes a run incomparable with every "
                "recorded number.")

    @classmethod
    def coerce(cls, value: Any, dotted: str = "prompting.bbox_event_grouping"):
        return _coerce(cls, value, dotted)


@dataclass
class SideExtend:
    band: int = 1
    pad: float = 0.25

    def validate(self) -> None:
        return None

    @classmethod
    def coerce(cls, value: Any, dotted: str = "side"):
        return _coerce(cls, value, dotted)


@dataclass
class BoundaryExtendParams:
    """`prompting.bbox_boundary_extend`. OFF, and it raises if turned on.

    Not ported because nothing here needs it yet and because `source:
    prediction` would make the feature cache unkeyable — the window would depend
    on model weights that change every step.
    """

    enabled: bool = False
    source: str = "polygon"
    bands: int = 10
    left: SideExtend = None       # type: ignore[assignment]
    top: SideExtend = None        # type: ignore[assignment]
    right: SideExtend = None      # type: ignore[assignment]
    bottom: SideExtend = None     # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.left is None:
            self.left = SideExtend(1, 0.25)
        if self.top is None:
            self.top = SideExtend(1, 0.25)
        if self.right is None:
            self.right = SideExtend(1, 0.25)
        if self.bottom is None:
            self.bottom = SideExtend(0, 0.0)

    def validate(self) -> None:
        if self.source not in ("polygon", "prediction"):
            raise ValueError(
                "prompting.bbox_boundary_extend.source must be 'polygon' or "
                f"'prediction'; got {self.source!r}")
        if self.enabled:
            raise NotImplementedError(
                "prompting.bbox_boundary_extend.enabled: true is not ported. "
                "At bands 10 / band 1 it fires on 90% of prompts, i.e. it is a "
                "constant directional pad rather than a trigger, and "
                "`source: prediction` additionally makes the feature cache "
                "unkeyable — the window would depend on weights that change "
                "every step.")

    @classmethod
    def coerce(cls, value: Any, dotted: str = "prompting.bbox_boundary_extend"):
        return _coerce(cls, value, dotted,
                       nested={"left": SideExtend, "top": SideExtend,
                               "right": SideExtend, "bottom": SideExtend})
