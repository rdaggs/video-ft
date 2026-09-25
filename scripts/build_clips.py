"""Build (or diff) a clip manifest. No GPU.

`repro.sh manifest` is the reason this is a script and not a function: a run
directory has to be able to prove its own corpus without importing anything the
run imported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smokeftv.clips import clip_signature, manifest_row       # noqa: E402
from smokeftv.config import load_config                        # noqa: E402
from smokeftv.corpus import build_clips                        # noqa: E402


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def diff(a: Path, b: Path) -> int:
    rows_a, rows_b = _read(a), _read(b)
    by_a = {r["clip_id"]: r for r in rows_a}
    by_b = {r["clip_id"]: r for r in rows_b}
    only_a = sorted(set(by_a) - set(by_b))
    only_b = sorted(set(by_b) - set(by_a))
    changed = [k for k in sorted(set(by_a) & set(by_b)) if by_a[k] != by_b[k]]
    if not (only_a or only_b or changed):
        print(f"clip manifests identical: {len(rows_a)} clips")
        return 0
    print(f"clip manifests DIFFER  ({a.name}: {len(rows_a)}, {b.name}: {len(rows_b)})",
          file=sys.stderr)
    for label, keys in (("only in " + a.name, only_a), ("only in " + b.name, only_b),
                        ("changed", changed)):
        if keys:
            print(f"  {label}: {len(keys)}", file=sys.stderr)
            for key in keys[:5]:
                print(f"    {key}", file=sys.stderr)
    print("\nThe corpus is supposed to be a pure function of (config, data, seed). "
          "A dirty\ndiff means annotations.json or polygons.coco.json changed on "
          "disk since the run.", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("A", "B"))
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    if args.diff:
        return diff(*args.diff)
    if not args.config or not args.out:
        parser.error("--config and --out are required unless --diff is given")

    cfg = load_config(args.config, args.overrides)
    names = {"train": cfg.data.train_incidents, "val": cfg.data.val_incidents,
             "test": cfg.data.test_incidents}[args.split]
    clips, _ = build_clips(cfg, names, args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for clip in clips:
            handle.write(json.dumps(manifest_row(clip, args.split)) + "\n")
    print(f"{len(clips)} clips -> {args.out}  (signature {clip_signature(clips)[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
