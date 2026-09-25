"""Prompt-dropout statistics over the real corpus. No GPU, no network.

`longest gap` is the number that matters: it is how many consecutive frames
memory has to carry the object with no box at all. If it is 0, the memory bank
is decorative and the run is a per-frame segmenter with extra steps.
"""

from __future__ import annotations

from scripts._probe_common import incidents_for, parse, rule
from smokeftv.config import load_config
from smokeftv.corpus import build_clips
from smokeftv.prompting import frame_keep_mask, keep_rate, schedule_stats


def main() -> int:
    args = parse(__doc__)
    cfg = load_config(args.config, args.overrides)
    names = [n for n, _ in incidents_for(cfg, args.split)
             if not args.incident or args.incident in n]
    clips, _ = build_clips(cfg, names, args.split)
    sched = cfg.train.dropout

    rule(f"prompt dropout over {len(clips)} {args.split} clips "
         f"(p ~ U({sched.p_min}, {sched.p_max}), warmup {sched.warmup_epochs} "
         f"epoch(s), length {cfg.data.clips.length})")
    print("  p is a KEEP rate. Frame 0 is never dropped.\n")
    print(f"  {'epoch':>5} {'p range':>12} {'unprompted':>11} {'mean gap':>9} "
          f"{'p90 gap':>8} {'max gap':>8} {'fully prompted':>15}")
    for epoch in range(cfg.train.epochs):
        low, high = keep_rate(sched, epoch)
        masks = [frame_keep_mask(len(c.frame_indices), sched, epoch,
                                 cfg.data.clips.seed, c.clip_id)[0]
                 for c in clips]
        s = schedule_stats(masks)
        tag = " (warmup)" if epoch < sched.warmup_epochs else ""
        print(f"  {epoch:>5} {f'U({low:.1f},{high:.1f})':>12} "
              f"{s['unprompted_frac']:>11.3f} {s['mean_longest_gap']:>9.2f} "
              f"{s['p90_longest_gap']:>8} {s['max_longest_gap']:>8} "
              f"{s['fully_prompted_frac']:>15.3f}{tag}")

    rule("  determinism")
    a = frame_keep_mask(8, sched, 3, cfg.data.clips.seed, clips[0].clip_id)
    b = frame_keep_mask(8, sched, 3, cfg.data.clips.seed, clips[0].clip_id)
    print(f"    same (epoch, clip, seed) -> same mask: {a == b}")
    print("    The schedule is seeded by identity, not stream position, so it")
    print("    reproduces from prompt_schedule.jsonl regardless of visit order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
