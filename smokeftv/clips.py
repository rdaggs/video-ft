"""Clip sampling: which 8-frame windows of which masklets the model trains on.

This module defines the corpus, so anything here is a resume tripwire — the
clip list is written to `ckpts/<run>/clip_manifest.jsonl` and
`checkpoint.check_resume` refuses when its signature moves.

Two decisions carry most of the weight.

**Stride.** Eight *consecutive* frames of slowly-evolving smoke is a
copy-the-last-mask task and teaches memory nothing, so clips are strided: at
`stride: 6` an 8-frame clip spans 43 real frames. But the median train incident
has only 40 labelled frames, so the configured stride alone produces clips for
just 69 of 144 incidents and silently discards half the corpus. Hence
`stride_clamp_to_fit`, which lowers the stride per incident until a clip fits:
1268 clip starts across **all** 144 incidents, 69 keeping the full span and 12
falling to stride 1. The realised stride is recorded per clip.

**Determinism.** `clip_rng` is seeded by `(seed, incident, masklet_id)`, not by
position in a stream. Position-seeded jitter would make the corpus a function of
the order incidents were iterated in, so one new folder at the top of the
listing would reshuffle every clip below it and `clip_manifest.jsonl` would stop
being reproducible from `config_resolved.yaml` alone.

Stdlib only.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from smokeftv.boxes import BoxLTRB, _coerce, _positive, box_area, box_union


@dataclass
class ClipParams:
    """The `clips:` block."""

    length: int = 8
    stride: int = 6
    stride_jitter: int = 2
    # SAM 2 uses 0.5. Reversal inverts the physical prior for a plume growing
    # monotonically off a fixed base. Ablation, not a default.
    reverse_prob: float = 0.0
    # Matches the eval's cap, so training and scoring see the same horizon.
    frame_cap: int = 60
    max_masklets_per_clip: int = 1
    stride_clamp_to_fit: bool = True
    seed: int = 20260924

    def validate(self) -> None:
        d = "clips"
        _positive("length", self.length, 2.0, dotted=d)
        _positive("stride", self.stride, 1.0, dotted=d)
        _positive("stride_jitter", self.stride_jitter, 0.0, dotted=d)
        _positive("reverse_prob", self.reverse_prob, 0.0, 1.0, dotted=d)
        _positive("frame_cap", self.frame_cap, 0.0, dotted=d)
        _positive("seed", self.seed, 0.0, dotted=d)
        if self.stride_jitter >= self.stride:
            raise ValueError(
                f"{d}.stride_jitter must be < {d}.stride, or a draw can reach "
                f"stride 0 and a clip becomes eight copies of one frame; got "
                f"jitter {self.stride_jitter} against stride {self.stride}")
        if self.max_masklets_per_clip != 1:
            raise NotImplementedError(
                f"{d}.max_masklets_per_clip is {self.max_masklets_per_clip}, but "
                "only 1 is implemented. There is no two-plume ground truth in "
                "this corpus to put in the second slot — annotations.json has at "
                "most one box per frame in 519 of 529 incidents, and the "
                "multi-instance frames in polygons.coco.json are wisps of one "
                "plume, which is why targets.gt_merge is union. See divergence "
                "`no_two_plume`.")
        if not isinstance(self.stride_clamp_to_fit, bool):
            raise ValueError(f"{d}.stride_clamp_to_fit must be true or false")

    @classmethod
    def coerce(cls, value: Any, dotted: str = "clips"):
        return _coerce(cls, value, dotted)


def capped_frames(frames: Sequence[int], cap: int) -> list[int]:
    """The first `cap` frames of polygon appearance, by frame index.

    Model-independent by construction — it reads GT presence and orders by frame
    index — so training and both arms of a paired comparison keep the same
    frames. An incident at or under the cap is returned whole. Verified to
    reproduce the eval's 2548 -> 1412 split on the GT test set.
    """
    ordered = sorted(frames)
    return ordered[:cap] if cap and cap > 0 else ordered


def clip_rng(seed: int, incident: str, masklet_id: int) -> random.Random:
    """A stream seeded by identity rather than by position. See module docstring."""
    digest = hashlib.sha256(f"{seed}:{incident}:{masklet_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def realised_stride(n_frames: int, params: ClipParams) -> int:
    """The largest stride <= `clips.stride` that still fits one clip.

    Returns `clips.stride` unchanged when the masklet is long enough, so the 69
    incidents that can afford the full 43-frame span keep it. Without the clamp
    the other 75 produce nothing at all.
    """
    if not params.stride_clamp_to_fit:
        return int(params.stride)
    if n_frames < params.length:
        return 0
    return max(1, min(int(params.stride), (n_frames - 1) // (params.length - 1)))


@dataclass(frozen=True)
class Clip:
    incident: str
    camera: str
    masklet_id: int
    image_hw: tuple[int, int]
    frame_indices: tuple[int, ...]
    stride: int
    reversed_: bool
    window: tuple[int, int, int, int]
    # Diagnostics recorded per clip so results can be broken down by how much
    # resolution the fixed window gave up. See crops.track_window.
    window_px: int
    window_ratio: float

    @property
    def clip_id(self) -> str:
        return f"{self.incident}:m{self.masklet_id}:f{self.frame_indices[0]}"


@dataclass
class ClipReport:
    """One line per incident for the startup log. `kept=0` beside a name is
    louder than the name quietly missing from a list."""

    incident: str
    polygon_frames: int = 0
    capped_frames: int = 0
    boxed_frames: int = 0
    masklets: int = 0
    stride: int = 0
    clips: int = 0
    note: str = ""

    def __str__(self) -> str:
        tail = f"  {self.note}" if self.note else ""
        return (f"{self.incident:46s} poly={self.polygon_frames:4d} "
                f"capped={self.capped_frames:3d} boxed={self.boxed_frames:3d} "
                f"masklets={self.masklets:2d} stride={self.stride:2d} "
                f"clips={self.clips:3d}{tail}")


def clip_starts(n_frames: int, params: ClipParams, stride: int,
                rng: random.Random) -> list[list[int]]:
    """Index positions (into the masklet's frame list) for each clip.

    Walks start offsets by `stride`, jittering each clip's own stride within
    `+/- stride_jitter` and clamping so the clip still fits. Positions, not frame
    indices, so the caller stays in charge of the frame list.
    """
    length = int(params.length)
    if n_frames < length or stride <= 0:
        return []
    out: list[list[int]] = []
    start = 0
    while True:
        jitter = rng.randint(-int(params.stride_jitter), int(params.stride_jitter))
        this = max(1, stride + jitter)
        span = (length - 1) * this
        while span > n_frames - 1 - start and this > 1:
            this -= 1
            span = (length - 1) * this
        if start + span > n_frames - 1:
            break
        out.append([start + i * this for i in range(length)])
        start += max(1, this)
    return out


def manifest_row(clip: Clip, split: str) -> dict:
    return {
        "clip_id": clip.clip_id,
        "split": split,
        "incident": clip.incident,
        "camera": clip.camera,
        "masklet_id": clip.masklet_id,
        "image_hw": list(clip.image_hw),
        "frame_indices": list(clip.frame_indices),
        "stride": clip.stride,
        "reversed": clip.reversed_,
        "window": list(clip.window),
        "window_px": clip.window_px,
        "window_ratio": round(clip.window_ratio, 4),
    }


def clip_signature(clips: Sequence[Clip]) -> str:
    """sha256 over the sorted clip ids AND their frame lists.

    The ids alone are not enough: `train_incidents: auto` means the same *number*
    of clips can be built from different frames tomorrow, and a count-only
    tripwire would not notice.
    """
    payload = "\n".join(
        f"{c.clip_id}|{','.join(str(f) for f in c.frame_indices)}"
        for c in sorted(clips, key=lambda c: c.clip_id))
    return hashlib.sha256(payload.encode()).hexdigest()
