"""Masklet identity: which boxes across frames are the same object.

`annotations.json` has per-frame boxes with no track ids, and their
frame-to-frame agreement is poor (median box-to-box IoU 0.731, 7.1% of
consecutive pairs below 0.3). So identity has to be derived, not read.

The linker is ported from `sam3-finetuned-cvf/smokeft/bbox_event_grouping.py` —
overlap OR edge-to-edge gap, with `max_frame_gap` expiry. Its union modes are a
prompt transform and stayed behind (see `smokeftv.boxes.EventGroupingParams`);
its *linking* step is exactly masklet assignment.

**But the linker does not produce identities on this corpus**, and the default
reflects that. With at most one box per frame in 519 of 529 incidents, a second
event means the chain BROKE, not that there are two plumes: `1022992_ec-245-0`
returns `[79, 4, 7, 5]` — one real track and three stubs — and
`1173104_ec-395-2`, whose `status.json` reads "segmented 2x smoke plume well",
returns `[10, 28]`, a break mid-incident. 45 of 145 train incidents look
"multi-object" this way and none of them are.

So under `masklets.one_masklet_per_incident` (the default) the masklet is the
incident's whole box chain, and the linker runs as a **fragmentation
diagnostic** in `scripts/probe_masklets.py`. Flip it off when per-plume masks
exist to make identity mean something.

Stdlib only, like `boxes.py`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from smokeftv.boxes import (
    BoxLTRB, _coerce, _positive, box_area, box_gap_frac, box_iou, box_union,
    boxes_from_annotations,
)


@dataclass
class MaskletParams:
    """The `masklets:` block."""

    link_iou: float = 0.1
    link_gap_frac: float = 0.5
    max_frame_gap: int = 5
    outlier_area_ratio: float = 8.0
    # The half-width, in frames, of the LOCAL median `kept()` compares against.
    # `outlier_area_ratio` is meaningless without it.
    outlier_window: int = 5
    # An event shorter than this is not a trackable object. Does NOT exist in
    # the reference linker — new here, and it gates the diagnostic.
    min_event_frames: int = 4
    one_masklet_per_incident: bool = True
    smokebase_crosscheck: bool = False

    def validate(self) -> None:
        d = "masklets"
        _positive("link_iou", self.link_iou, 0.0, 1.0, dotted=d)
        _positive("link_gap_frac", self.link_gap_frac, 0.0, dotted=d)
        _positive("max_frame_gap", self.max_frame_gap, 0.0, dotted=d)
        _positive("outlier_area_ratio", self.outlier_area_ratio, 1.0, dotted=d)
        _positive("outlier_window", self.outlier_window, 1.0, dotted=d)
        _positive("min_event_frames", self.min_event_frames, 1.0, dotted=d)
        for name in ("one_masklet_per_incident", "smokebase_crosscheck"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{d}.{name} must be true or false; "
                                 f"got {getattr(self, name)!r}")
        if self.smokebase_crosscheck:
            raise NotImplementedError(
                "masklets.smokebase_crosscheck: true — there is nothing to "
                "cross-check against. smokebase_labels.json exists for 26 of "
                "the 28 incidents under splits.gt_dataset_root (the HELD-OUT "
                "test set) and for zero incidents under shared.data_root. The "
                "detected fallback is one point per incident, which cannot "
                "cross-check a per-plume base count even where it exists.")

    @classmethod
    def coerce(cls, value: Any, dotted: str = "masklets"):
        return _coerce(cls, value, dotted)


@dataclass(frozen=True)
class Observation:
    frame_index: int
    slot: int
    box: BoxLTRB


@dataclass
class Masklet:
    masklet_id: int
    observations: list[Observation] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame_index

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame_index

    @property
    def last_box(self) -> BoxLTRB:
        return self.observations[-1].box

    @property
    def span(self) -> int:
        return self.last_frame - self.first_frame + 1

    def observed_frames(self) -> set[int]:
        return {o.frame_index for o in self.observations}

    def union(self) -> BoxLTRB:
        return box_union([o.box for o in self.observations])

    def median_area(self) -> float:
        return statistics.median([box_area(o.box) for o in self.observations])

    def kept(self, params: MaskletParams) -> list[Observation]:
        """Observations minus the area *spikes*. Never empty.

        Local, against the median of the observations within `outlier_window`
        frames on each side — not against the masklet's own median, which was
        the first version of this and was wrong in a way worth recording. A
        plume grows: `1151608_ec-219-4` runs from 928 px^2 to 489,996 px^2 over
        53 frames, so a masklet-wide median of 2944 px^2 called 22 of its 53
        boxes outliers and threw away the entire second half of the fire.
        Growth is the signal here. A local median tracks it and still catches
        what this is actually for — the single frame boxed around half the sky.

        Never consulted for a frame's own prompt: dropping an observation
        removes it from other frames' unions without ever substituting some
        other frame's box for a box this frame really drew.
        """
        if len(self.observations) < 5:
            # Too few for a local median to mean anything, and dropping one of
            # four would take a quarter of the masklet with it.
            return list(self.observations)
        span = max(int(params.outlier_window), 1)
        kept: list[Observation] = []
        for obs in self.observations:
            neighbours = [box_area(o.box) for o in self.observations
                          if o is not obs
                          and abs(o.frame_index - obs.frame_index) <= span]
            if len(neighbours) < 2:
                kept.append(obs)
                continue
            limit = statistics.median(neighbours) * float(params.outlier_area_ratio)
            if box_area(obs.box) <= limit:
                kept.append(obs)
        return kept or list(self.observations)


def _best_masklet(box: BoxLTRB, open_masklets: list[Masklet],
                  claimed: set[int], params: MaskletParams) -> Masklet | None:
    """The open masklet this box most likely continues, or None for a new one.

    Affinity is the **max** of IoU-against-last-box and IoU-against-running-union:
    a plume whose box walks from base to column overlaps *where it has been* even
    when it no longer overlaps *where it was last seen*. The gap is measured
    against `last_box` only. The test is OR, both inclusive, and the tie-break is
    lexicographic `(affinity, -gap)` — higher affinity first, then smaller gap.
    """
    scored: list[tuple[float, float, Masklet]] = []
    for masklet in open_masklets:
        if masklet.masklet_id in claimed:
            continue
        affinity = max(box_iou(box, masklet.last_box), box_iou(box, masklet.union()))
        gap = box_gap_frac(box, masklet.last_box)
        if affinity >= params.link_iou or gap <= params.link_gap_frac:
            scored.append((affinity, -gap, masklet))
    if not scored:
        return None
    return max(scored, key=lambda t: (t[0], t[1]))[2]


@dataclass
class MaskletGrouping:
    params: MaskletParams
    masklets: list[Masklet]
    image_hw: tuple[int, int] | None = None
    owner: dict[tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def build(cls, boxes_by_frame: Mapping[int, Sequence[BoxLTRB]],
              params: MaskletParams | None = None,
              image_hw: tuple[int, int] | None = None) -> "MaskletGrouping":
        """One greedy forward pass in frame order.

        Three properties the port must keep:
          * expiry is checked **once per frame, before the slot loop**, so a
            masklet created this frame cannot expire this frame, and `<=` means
            one exactly `max_frame_gap` frames stale is still open;
          * an expired masklet is dropped and **can never be resumed**;
          * `claimed` allows at most one box per masklet per frame, so two boxes
            on a frame are two masklets.

        `masklet_id = len(masklets)` — a dense 0-based counter in creation order,
        so `masklets[masklet_id]` indexes directly. Slot order is load-bearing:
        the clip builder pairs polygons to boxes by IoU over the frame's box
        list, and a reordered list silently re-pairs targets with prompts.
        """
        params = MaskletParams.coerce(params)
        masklets: list[Masklet] = []
        open_masklets: list[Masklet] = []
        owner: dict[tuple[int, int], int] = {}
        for frame_index in sorted(boxes_by_frame):
            open_masklets = [m for m in open_masklets
                             if frame_index - m.last_frame <= params.max_frame_gap]
            claimed: set[int] = set()
            for slot, raw in enumerate(boxes_by_frame[frame_index]):
                box = tuple(float(v) for v in raw)
                chosen = _best_masklet(box, open_masklets, claimed, params)
                if chosen is None:
                    chosen = Masklet(masklet_id=len(masklets))
                    masklets.append(chosen)
                    open_masklets.append(chosen)
                chosen.observations.append(
                    Observation(frame_index=frame_index, slot=slot, box=box))
                claimed.add(chosen.masklet_id)
                owner[(frame_index, slot)] = chosen.masklet_id
        return cls(params=params, masklets=masklets, image_hw=image_hw, owner=owner)

    @classmethod
    def from_annotations(cls, path: str | Path, params: MaskletParams | None = None,
                         image_hw: tuple[int, int] | None = None) -> "MaskletGrouping":
        return cls.build(boxes_from_annotations(path), params, image_hw)

    def masklet_of(self, frame_index: int, slot: int = 0) -> Masklet | None:
        mid = self.owner.get((frame_index, slot))
        return None if mid is None else self.masklets[mid]

    def masklets_at(self, frame_index: int) -> list[Masklet]:
        return [m for m in self.masklets if frame_index in m.observed_frames()]

    def viable(self) -> list[Masklet]:
        """Masklets long enough to be a trackable object.

        Counted on DISTINCT OBSERVED FRAMES of the raw observations, not on
        `kept()` and not on `span`. `kept()` is an area-spike filter that never
        empties, so counting it would make a length threshold depend on box
        areas; `span` counts frames the masklet was never seen on, so two boxes
        five frames apart would pass a threshold of four. This is a statement
        about presence.
        """
        return [m for m in self.masklets
                if len(m.observed_frames()) >= self.params.min_event_frames]

    def stats(self) -> dict:
        viable = self.viable()
        lengths = sorted(len(m.observed_frames()) for m in viable)
        return {
            "masklets": len(self.masklets),
            "viable": len(viable),
            "observations": sum(len(m.observations) for m in self.masklets),
            "median_length": statistics.median(lengths) if lengths else 0,
            "max_length": lengths[-1] if lengths else 0,
            "lengths": lengths,
        }


def incident_masklets(boxes_by_frame: Mapping[int, Sequence[BoxLTRB]],
                      params: MaskletParams | None = None,
                      image_hw: tuple[int, int] | None = None) -> list[Masklet]:
    """The masklets a clip builder should actually iterate.

    Under `one_masklet_per_incident` (the default) this is ONE masklet holding
    every box in the incident, regardless of what the linker says — see the
    module docstring for why the linker's extra events are broken chains rather
    than second plumes. The linker still runs in `probe_masklets.py`, where its
    fragmentation is the thing being reported.
    """
    params = MaskletParams.coerce(params)
    if not params.one_masklet_per_incident:
        return MaskletGrouping.build(boxes_by_frame, params, image_hw).viable()
    observations = [
        Observation(frame_index=frame_index, slot=slot, box=tuple(float(v) for v in box))
        for frame_index in sorted(boxes_by_frame)
        for slot, box in enumerate(boxes_by_frame[frame_index])
    ]
    if len(observations) < params.min_event_frames:
        return []
    return [Masklet(masklet_id=0, observations=observations)]
