"""Masklet diagnostics. No GPU, no network.

Prints what the linker makes of the corpus, and — because the linker does NOT
produce identities on this data — how badly it fragments. Read the fragmentation
table before believing any masklet count.
"""

from __future__ import annotations

import collections
import statistics

from scripts._probe_common import incidents_for, parse, rule
from smokeftv.boxes import boxes_from_annotations
from smokeftv.config import load_config
from smokeftv.incidents import ANNOTATIONS_FILENAME, smoke_descriptors
from smokeftv.masklets import MaskletGrouping, incident_masklets
from pathlib import Path


def main() -> int:
    args = parse(__doc__)
    cfg = load_config(args.config, args.overrides)
    params = cfg.data.masklets
    rows = [(n, s) for n, s in incidents_for(cfg, args.split)
            if not args.incident or args.incident in n]

    per_incident = collections.Counter()
    lengths: list[int] = []
    frag: list[tuple[str, int, list[int]]] = []
    two_box_frames = 0
    two_box_distinct = 0
    scanned = 0

    for name, _ in rows:
        path = Path(cfg.data.root) / name / ANNOTATIONS_FILENAME
        if not path.is_file():
            continue
        scanned += 1
        boxes = boxes_from_annotations(path)
        grouping = MaskletGrouping.build(boxes, params)
        viable = grouping.viable()
        per_incident[len(viable)] += 1
        lens = sorted(len(m.observed_frames()) for m in viable)
        lengths.extend(lens)
        if len(viable) >= 2:
            frag.append((name, len(viable), lens))
        for frame, bs in boxes.items():
            if len(bs) >= 2:
                two_box_frames += 1
                ids = {grouping.owner.get((frame, slot)) for slot in range(len(bs))}
                two_box_distinct += len(ids) >= 2

    rule(f"linker over {scanned} {args.split} incident(s)"
         f"  (link_iou {params.link_iou}, link_gap_frac {params.link_gap_frac}, "
         f"max_frame_gap {params.max_frame_gap}, min_event_frames {params.min_event_frames})")
    print("  events per incident:", dict(sorted(per_incident.items())))
    if lengths:
        lengths.sort()
        print(f"  event length: n={len(lengths)} min={lengths[0]} "
              f"median={statistics.median(lengths):.0f} "
              f"p90={lengths[int(0.9*(len(lengths)-1))]} max={lengths[-1]}")

    rule("fragmentation — incidents the linker splits")
    print(f"  {len(frag)} of {scanned} incidents return >=2 events.")
    print("  With <=1 box per frame these are BROKEN CHAINS, not second plumes:")
    print("  a long track beside stubs is the signature.\n")
    for name, k, lens in sorted(frag, key=lambda r: -r[1])[:12]:
        stub = sum(1 for n in lens if n < params.min_event_frames * 2)
        print(f"    {name:46s} {k} events {lens}"
              f"{'   <- ' + str(stub) + ' stub(s)' if stub else ''}")

    rule("simultaneous boxes — the only real multi-object signal")
    print(f"  frames with >=2 boxes: {two_box_frames}")
    print(f"  of those, assigned to distinct masklets: {two_box_distinct}")

    rule("multi-plume incidents, by the labeler's own words")
    import re
    pattern = re.compile(r"2x|two |second plume|multiple|both plume|twin|dual", re.I)
    found = 0
    for name, split in incidents_for(cfg, "all"):
        notes = smoke_descriptors(cfg.data.root, name)
        if any(pattern.search(t) for t in notes):
            found += 1
            print(f"    [{split:5s}] {name:46s} {notes}")
    print(f"\n  {found} corpus-wide. This is why clips.max_masklets_per_clip is 1")
    print("  and why the identity metrics are out of scope — see divergence")
    print("  `no_two_plume`.")

    rule(f"effective masklets (one_masklet_per_incident="
         f"{params.one_masklet_per_incident})")
    counts = collections.Counter()
    for name, _ in rows:
        path = Path(cfg.data.root) / name / ANNOTATIONS_FILENAME
        if path.is_file():
            counts[len(incident_masklets(boxes_from_annotations(path), params))] += 1
    print("  masklets per incident:", dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
