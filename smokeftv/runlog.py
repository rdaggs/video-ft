"""Run artifacts: logging, CSV/JSONL, provenance, splits, and repro.sh.

`ckpts/<run>/` must contain everything needed to recreate the run. Two rules
carry that:

* **`config_resolved.yaml` is written first**, before a single JPEG is opened,
  so a run that dies in the corpus scan still records what it was trying to do.
* **Nothing here writes a number anyone quotes.** The artifact you quote is the
  report; these files are what let you believe it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROVENANCE_SCHEMA = 1


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path, short: bool = False) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:16] if short else digest


def setup_logging(run_dir: Path, name: str = "smokeftv") -> logging.Logger:
    """A logger that writes to both stdout and `<run_dir>/train.log`.

    `train.log` and the `run.log` symlink coexist and say different things:
    `run.log` is train.sh's stdout — banner included — and `train.log` is the
    logger alone, which is what a plotting script wants to grep.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    # Do not propagate to root. Importing sam3 pulls in libraries that configure
    # the root logger, and every line would otherwise appear twice — once
    # formatted and once as `INFO:smokeftv:...`.
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(run_dir / "train.log")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


class CsvLog:
    """Append-only CSV whose header is fixed by the first row it ever writes.

    Fixed from the file on disk when one exists, so a resumed run's CSV stays
    parseable. A column added only to later rows would be dropped by
    `extrasaction="ignore"` rather than corrupting the file.
    """

    def __init__(self, path: Path):
        self.path = path
        self.fields: list[str] | None = None
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), None)
            if header:
                self.fields = header

    def write(self, row: dict) -> None:
        new = self.fields is None
        if new:
            self.fields = list(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields,
                                    extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(row)


class JsonlLog:
    """One JSON object per line, flushed. Carries nested values a CSV cannot."""

    def __init__(self, path: Path, append: bool = True):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a" if append else "w", encoding="utf-8")

    def write(self, row: dict) -> None:
        self.handle.write(json.dumps(row, sort_keys=False) + "\n")
        self.handle.flush()

    def write_all(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.handle.write(json.dumps(row, sort_keys=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def git_info(repo_root: Path) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(args, cwd=repo_root, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {
        "sha": run("git", "rev-parse", "--short", "HEAD") or "unknown",
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def env_fingerprint() -> dict:
    out: dict[str, Any] = {
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "user": os.environ.get("USER", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            out["gpu_name"] = torch.cuda.get_device_name(0)
            out["gpu_count"] = torch.cuda.device_count()
    except Exception:                                   # noqa: BLE001
        out["torch"] = None
    for module in ("cv2", "numpy"):
        try:
            out[module] = __import__(module).__version__
        except Exception:                               # noqa: BLE001
            out[module] = None
    return out


def code_fingerprints(package_dir: Path) -> dict:
    """path + sha + bytes for every module, so "the same experiment" is checkable."""
    return {p.name: {"sha256": sha256_file(p, short=True), "bytes": p.stat().st_size}
            for p in sorted(package_dir.glob("*.py"))}


def write_splits_resolved(path: Path, cfg) -> dict:
    """The camera-level decision the incident names encode.

    Separate from `config_resolved.yaml`, which records the *names*. This one
    records what those names were chosen to do, which is the thing a reader of a
    report needs and the thing that silently rots.
    """
    from smokeftv.incidents import camera_of
    import yaml
    cams = lambda names: sorted({camera_of(n) for n in names})
    held = set(cams(cfg.data.val_incidents)) | set(cams(cfg.data.test_incidents))
    blob = {
        "data_root": cfg.data.root,
        "gt_dataset_root": cfg.data.gt_dataset_root,
        "train_incidents": list(cfg.data.train_incidents),
        "train_cameras": cams(cfg.data.train_incidents),
        "val_incidents": list(cfg.data.val_incidents),
        "val_cameras": cams(cfg.data.val_incidents),
        "test_incidents": list(cfg.data.test_incidents),
        "test_cameras": cams(cfg.data.test_incidents),
        "held_out_cameras": sorted(held),
        "leaked_cameras": sorted(set(cams(cfg.data.train_incidents)) & held),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(blob, sort_keys=False), encoding="utf-8")
    return blob


def write_provenance(path: Path, **fields) -> dict:
    blob = {"schema": PROVENANCE_SCHEMA, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    return blob


def update_provenance(path: Path, **fields) -> None:
    """Only `status` and `finished_utc` mutate. A `status: running` on a run
    whose log stopped is the died-before-the-handler signal."""
    if not path.is_file():
        return
    blob = json.loads(path.read_text(encoding="utf-8"))
    blob.update(fields)
    path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")


REPRO_TEMPLATE = """\
#!/usr/bin/env bash
# Regenerate this run's corpus, or rerun it. Written by scripts/train.py.
#
# `repro.sh manifest` rebuilds the clip list from this run's OWN
# config_resolved.yaml and diffs it against what the run actually built. A clean
# diff means the corpus is still a function of (config, data, seed) and nothing
# else. The only thing that can make it dirty is annotations.json or
# polygons.coco.json changing on disk — which is exactly what it is for.
set -euo pipefail

REPO={repo_root}
RUN={run_name}
TAG={tag}
GIT_SHA={git_sha}
cd "$REPO"

HERE="{ckpt_dir}"
NOW="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
[[ "$NOW" == "$GIT_SHA" ]] || \\
  echo "repro.sh: repo is at $NOW, this run was $GIT_SHA" >&2

case "${{1:-manifest}}" in
  manifest)
    .venv/bin/python -m scripts.build_clips \\
      --config "$HERE/config_resolved.yaml" \\
      --split train --out "$HERE/repro_manifest.jsonl"
    .venv/bin/python -m scripts.build_clips --diff \\
      "$HERE/clip_manifest.jsonl" "$HERE/repro_manifest.jsonl"
    ;;
  train)
    ./train.sh --gpu "${{2:?usage: repro.sh train <gpu>}}" --tag "$TAG" \\
      --config "$HERE/config_resolved.yaml" \\
      > "logs/$(date -u +%Y%m%d_%H%M%SZ)_${{TAG}}-repro.log" 2>&1
    ;;
  *) echo "usage: repro.sh [manifest|train <gpu>]" >&2; exit 2 ;;
esac
"""


def write_repro(path: Path, *, repo_root: Path, run_name: str, tag: str,
                git_sha: str, ckpt_dir: Path) -> None:
    path.write_text(REPRO_TEMPLATE.format(
        repo_root=repo_root, run_name=run_name, tag=tag, git_sha=git_sha,
        ckpt_dir=ckpt_dir), encoding="utf-8")
    path.chmod(0o755)
