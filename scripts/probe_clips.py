"""The clip corpus a run will actually train on. No GPU, no network.

Two things to read before trusting a training run: the realised-stride
histogram (how many incidents had to give up the configured stride to produce a
clip at all) and the window_ratio table (how much resolution the fixed window
costs against image mode's per-frame window).
"""

from __future__ import annotations

import collections
import statistics

from scripts._probe_common import incidents_for, parse, rule
from smokeftv.clips import clip_signature
from smokeftv.config import load_config
from smokeftv.corpus import build_clips


def _pct(values, q):
    values = sorted(values)
    return values[int(q * (len(values) - 1))] if values else 0


def main() -> int:
    args = parse(__doc__)
    cfg = load_config(args.config, args.overrides)
    params = cfg.data.clips
    splits = ["train", "val"] if args.split == "all" else [args.split]

    for split in splits:
        names = [n for n, _ in incidents_for(cfg, split)
                 if not args.incident or args.incident in n]
        clips, reports = build_clips(cfg, names, split)

        rule(f"{split}: {len(clips)} clips from {len(names)} incidents "
             f"(length {params.length}, stride {params.stride}"
             f"+/-{params.stride_jitter}, frame_cap {params.frame_cap}, "
             f"seed {params.seed})")
        contributing = sum(1 for r in reports if r.clips)
        print(f"  incidents contributing clips : {contributing}/{len(reports)}")
        print(f"  polygon frames               : {sum(r.polygon_frames for r in reports)}")
        print(f"  after frame_cap              : {sum(r.capped_frames for r in reports)}")
        print(f"  clip signature               : {clip_signature(clips)[:16]}")

        strides = collections.Counter(c.stride for c in clips)
        rule("  realised stride — the per-incident clamp")
        span = lambda s: (params.length - 1) * s + 1
        for stride in sorted(strides):
            n = strides[stride]
            flag = "  <- full configured span" if stride == params.stride else ""
            print(f"    stride {stride}  spans {span(stride):3d} frames  "
                  f"{n:5d} clips ({100*n/max(len(clips),1):4.1f}%){flag}")
        print(f"\n    Without the clamp only incidents with >={span(params.stride)} "
              "usable frames\n    would produce anything at all.")

        rule("  window_ratio — resolution given up to the fixed window")
        ratios = [c.window_ratio for c in clips]
        pxs = [c.window_px for c in clips]
        if ratios:
            print(f"    ratio vs per-frame window: median {statistics.median(ratios):.2f}x  "
                  f"p75 {_pct(ratios,.75):.2f}x  p90 {_pct(ratios,.90):.2f}x  "
                  f"max {max(ratios):.2f}x")
            print(f"    window side px           : median {statistics.median(pxs):.0f}  "
                  f"p90 {_pct(pxs,.90)}  max {max(pxs)}")
            short = min(c.image_hw[0] for c in clips)
            sat = sum(1 for c in clips if c.window_px >= short)
            print(f"    clips saturating the {short}px short side: {sat} "
                  f"({100*sat/len(clips):.1f}%)")
            print(f"\n    Both windows resize to crop.image_size ({cfg.crop.image_size}), "
                  "so a ratio of\n    2.0 means roughly a quarter of the pixels on "
                  "the plume. That is the\n    price of spatial registration; it belongs "
                  "in the report, not in a\n    footnote.")

        empty = [r for r in reports if not r.clips]
        if empty:
            rule(f"  {len(empty)} incident(s) contributed nothing")
            for r in empty[:10]:
                print(f"    {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
