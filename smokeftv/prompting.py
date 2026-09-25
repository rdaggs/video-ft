"""The per-frame prompt schedule: which frames keep their box.

A box is available on every frame, matching inference. During training it is
withheld on a random subset so memory attention is load-bearing — without the
dropout the shortest path to low loss runs entirely through the prompt encoder,
memory gets no gradient, and you ship a per-frame segmenter carrying a
decorative memory bank.

Two rules the rest of the pipeline depends on:

* **Frame 0 always keeps its box.** It is the conditioning frame, and a clip
  with no prompt at all has nothing to track.
* **The mask gates the prompt-encoder input ONLY.** The crop window is built
  from the clip's boxes before any draw happens (see `crops.windows_for_clip`),
  because if a dropped frame also lost its crop the model could identify dropped
  frames from geometry alone and the dropout would teach it nothing.

Stdlib only.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

DETECTOR_DUMP_SPEC = """\
<detector_boxes_path>/<incident>.json, one object per incident:
  {"incident": "<folder name>",
   "frames": [{"frame_index": 12,
               "boxes": [{"left": 0.0, "top": 0.0,
                          "width": 0.0, "height": 0.0, "score": 0.0}]}]}
Same left/top/width/height keys, the same full-frame pixel coordinates and the
same frames-with-no-boxes-are-omitted rule as annotations.json, so
boxes.boxes_from_annotations reads it unchanged."""


def boxes_for_source(prompt_cfg, root: str | Path, incident: str) -> dict:
    """The prompt boxes for an incident, per `prompting.box_source`."""
    from smokeftv.boxes import boxes_from_annotations
    from smokeftv.incidents import ANNOTATIONS_FILENAME
    if prompt_cfg.box_source == "annotations":
        return boxes_from_annotations(Path(root) / incident / ANNOTATIONS_FILENAME)
    raise NotImplementedError(
        f"prompting.box_source: {prompt_cfg.box_source!r} is not implemented.\n"
        f"{DETECTOR_DUMP_SPEC}\n"
        "`detector` is the one that closes the train/test gap — the detector's "
        "real failure modes (one box spanning both plumes, a box locked on the "
        "dense core, drift onto shadow or steam, outright misses) are the "
        "training signal — but no dump exists, and changing this invalidates "
        "comparison with every recorded number. See divergence "
        "`detector_prompts`.")


def prompt_rng(seed: int, epoch: int, clip_id: str) -> random.Random:
    """Seeded by identity, not stream position, for the same reason
    `clips.clip_rng` is: so the schedule is reproducible from the run record
    regardless of the order clips were visited in."""
    digest = hashlib.sha256(f"{seed}:{epoch}:{clip_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def keep_rate(schedule, epoch: int) -> tuple[float, float]:
    """The (p_min, p_max) this epoch draws from.

    During `warmup_epochs` the rate is pinned at `p_max` — the DENSEST prompting
    — so early epochs still get a strong signal while the memory modules are
    moving fastest. After that it is the full range.
    """
    if epoch < int(schedule.warmup_epochs):
        return float(schedule.p_max), float(schedule.p_max)
    return float(schedule.p_min), float(schedule.p_max)


def frame_keep_mask(length: int, schedule, epoch: int, seed: int,
                    clip_id: str) -> tuple[list[bool], float]:
    """`(keep_mask, p)` for one clip. `keep_mask[0]` is always True."""
    rng = prompt_rng(seed, epoch, clip_id)
    low, high = keep_rate(schedule, epoch)
    p = rng.uniform(low, high)
    mask = [True] + [rng.random() < p for _ in range(max(length - 1, 0))]
    return mask, p


def schedule_row(epoch: int, clip_id: str, p: float, mask: list[bool]) -> dict:
    return {"epoch": epoch, "clip_id": clip_id, "p": round(p, 6),
            "keep": [bool(v) for v in mask], "n_kept": int(sum(mask))}


def schedule_stats(masks) -> dict:
    """Corpus-level dropout statistics, for `probe_schedule.py`.

    `longest_gap` is the thing that matters: it is how many consecutive frames
    memory has to carry the object alone.
    """
    total = kept = 0
    gaps: list[int] = []
    fully = 0
    for mask in masks:
        total += len(mask)
        kept += sum(mask)
        fully += all(mask)
        run = longest = 0
        for keep in mask:
            run = 0 if keep else run + 1
            longest = max(longest, run)
        gaps.append(longest)
    gaps.sort()
    n = len(gaps) or 1
    return {
        "clips": len(gaps),
        "unprompted_frac": round(1 - kept / max(total, 1), 4),
        "mean_longest_gap": round(sum(gaps) / n, 3),
        "p90_longest_gap": gaps[int(0.9 * (n - 1))] if gaps else 0,
        "max_longest_gap": gaps[-1] if gaps else 0,
        "fully_prompted_frac": round(fully / n, 4),
    }
