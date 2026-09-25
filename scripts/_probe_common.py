"""Shared argument handling for the no-GPU probes.

All three read `configs/train_video.yaml` through `load_config`, so they see
exactly the values a run will — a probe that reads the snapshot directly would
miss anything the run config overrides.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse(description: str, *, splits: bool = True):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", "--config", type=Path,
                        default=REPO_ROOT / "configs" / "train_video.yaml")
    if splits:
        parser.add_argument("--split", default="train",
                            choices=["train", "val", "test", "all"])
    parser.add_argument("--incident", default=None,
                        help="restrict to incidents containing this substring")
    parser.add_argument("overrides", nargs="*", help="dotted config overrides")
    return parser.parse_args()


def incidents_for(cfg, split: str) -> list[tuple[str, str]]:
    table = {"train": cfg.data.train_incidents, "val": cfg.data.val_incidents,
             "test": cfg.data.test_incidents}
    if split == "all":
        return [(name, s) for s in ("train", "val", "test") for name in table[s]]
    return [(name, split) for name in table[split]]


def rule(title: str, width: int = 96) -> None:
    print(f"\n{title}\n{'-' * min(len(title), width)}")
