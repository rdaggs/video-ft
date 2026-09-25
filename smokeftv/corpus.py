"""Assembling the clip corpus: masklets -> windows -> clips.

This is the module that turns a `Config` into the list of samples a run trains
on, and therefore the module whose output is a resume tripwire. Everything here
is a pure function of (config, data on disk, `clips.seed`) — no model, no GPU,
no wall clock — which is what makes `repro.sh` able to regenerate an identical
`clip_manifest.jsonl`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from smokeftv import crops
from smokeftv.boxes import NO_BOX_GAP, POLYGON_BOX_GAP, box_area, resolve_box
from smokeftv.clips import (
    Clip, ClipParams, ClipReport, capped_frames, clip_rng, clip_starts,
    realised_stride,
)
from smokeftv.incidents import camera_of, frame_path, load_polygons
from smokeftv.masklets import incident_masklets
from smokeftv.prompting import boxes_for_source, jitter_boxes

DEFAULT_IMAGE_HW = (1872, 3328)


def _window_for(cfg, boxes_by_frame, frames: Sequence[int],
                image_hw) -> tuple[int, int, int, int]:
    """The crop every frame of this masklet is encoded under.

    Under `window_scope: masklet` this is the union of the whole track's boxes,
    which is what eval sees when it propagates the masklet end to end. Under
    `clip` it would be the clip's own union — tighter, but it moves with
    `stride_jitter`, so `config._check_cache` refuses to combine it with a live
    feature cache.
    """
    boxes = [b for f in frames for b in boxes_by_frame.get(f, ())]
    if cfg.crop.mode == "full_frame" or not boxes:
        return (0, 0, int(image_hw[1]), int(image_hw[0]))
    if cfg.crop.mode == "per_frame_box":
        return crops.crop_for_box(
            boxes[0], image_hw, pad_frac=cfg.crop.crop_pad_frac,
            pad_top_extra=cfg.crop.crop_pad_top_extra,
            bbox_pad=cfg.crop.bbox_pad, square_infer=cfg.crop.square_infer)
    return crops.track_window(boxes, image_hw, union_pad=cfg.crop.union_pad,
                              square=cfg.crop.square_infer)


def _window_ratio(window, boxes) -> float:
    """window side / the median per-frame window side.

    Recorded per clip so results can be broken down by how much resolution the
    fixed window gave up. 1.0 means it cost nothing; the corpus median is ~1.6.
    """
    if not boxes:
        return 1.0
    import statistics
    side = max(window[2] - window[0], window[3] - window[1], 1)
    per_frame = [max(b[2] - b[0], b[3] - b[1], 1.0) * 1.5 for b in boxes]
    return side / max(statistics.median(per_frame), 1.0)


def build_incident_clips(cfg, incident: str, split: str
                         ) -> tuple[list[Clip], ClipReport]:
    """Every clip this incident contributes, plus the line it logs."""
    root = cfg.data.root
    params: ClipParams = cfg.data.clips
    report = ClipReport(incident=incident)

    polygons, image_hw = load_polygons(root, incident)
    image_hw = image_hw or DEFAULT_IMAGE_HW
    report.polygon_frames = len(polygons)
    if not polygons:
        report.note = "no polygons"
        return [], report

    boxes_by_frame = boxes_for_source(cfg.prompt, root, incident)
    report.boxed_frames = len(boxes_by_frame)

    masklets = incident_masklets(boxes_by_frame, cfg.data.masklets, image_hw)
    report.masklets = len(masklets)
    if not masklets:
        report.note = "no viable masklet"
        return [], report

    # The horizon, applied to GT presence and ordered by frame index — the same
    # rule the eval's frame cap uses, so training and scoring see one horizon.
    kept = capped_frames(sorted(polygons), params.frame_cap)
    report.capped_frames = len(kept)

    box_frames = sorted(boxes_by_frame)
    # Linked above from the clean boxes so jitter cannot change which samples
    # exist; the window is built from the boxes the prompts will actually carry.
    prompt_boxes = jitter_boxes(boxes_by_frame, cfg.prompt.box_jitter, incident,
                                image_hw)
    out: list[Clip] = []
    for masklet in masklets:
        observed = masklet.observed_frames()
        # A frame is usable when it has a target AND a prompt we can reach —
        # its own box, or one carried forward within bbox_propagate_frames.
        usable = []
        for frame in kept:
            if frame < masklet.first_frame or frame > masklet.last_frame:
                if frame not in observed:
                    continue
            resolved = resolve_box(frame, boxes_by_frame,
                                   cfg.prompt.bbox_propagate_frames, box_frames)
            if resolved is None and not cfg.prompt.bbox_from_polygon:
                continue
            if not frame_path(root, incident, frame).is_file():
                continue
            usable.append(frame)
        if len(usable) < params.length:
            report.note = f"masklet {masklet.masklet_id}: {len(usable)} usable frames"
            continue

        stride = realised_stride(len(usable), params)
        report.stride = max(report.stride, stride)
        rng = clip_rng(params.seed, incident, masklet.masklet_id)
        window = _window_for(cfg, prompt_boxes, usable, image_hw)
        ratio = _window_ratio(window, [b for f in usable
                                       for b in prompt_boxes.get(f, ())])
        side = max(window[2] - window[0], window[3] - window[1])

        for positions in clip_starts(len(usable), params, stride, rng):
            frames = tuple(usable[i] for i in positions)
            reversed_ = rng.random() < params.reverse_prob
            if reversed_:
                frames = tuple(reversed(frames))
            out.append(Clip(
                incident=incident, camera=camera_of(incident),
                masklet_id=masklet.masklet_id, image_hw=image_hw,
                frame_indices=frames, stride=stride, reversed_=reversed_,
                window=window, window_px=int(side), window_ratio=float(ratio)))

    report.clips = len(out)
    return out, report


def build_clips(cfg, incidents: Sequence[str], split: str
                ) -> tuple[list[Clip], list[ClipReport]]:
    """The corpus for one split, in sorted `clip_id` order."""
    clips: list[Clip] = []
    reports: list[ClipReport] = []
    for incident in incidents:
        got, report = build_incident_clips(cfg, incident, split)
        clips.extend(got)
        reports.append(report)
    clips.sort(key=lambda c: c.clip_id)
    return clips, reports
