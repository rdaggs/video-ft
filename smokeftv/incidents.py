"""Corpus discovery, the camera rule, and reading an incident's labels.

Stdlib only. `config.py` imports `auto_train_incidents` from here and that is
the single filesystem read the config layer performs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

POLYGONS_FILENAME = "polygons.coco.json"
ANNOTATIONS_FILENAME = "annotations.json"
STATUS_FILENAME = "status.json"
FRAMES_DIRNAME = "frames"
AUTO = "auto"


def camera_of(incident: str) -> str:
    """The `ec-<N>` token. Splits hold out whole cameras, not whole incidents.

    Two incidents on one camera share a horizon, a site and usually an
    overlapping field of view, so holding out an incident while training on its
    neighbour is not a held-out measurement. The token is the middle field of
    `<detection id>_ec-<camera>-<n>_<start>_<end>`; note it is NOT the leading
    detection id, which is per-incident.
    """
    match = re.search(r"_ec-(\d+)-\d+(?:_|$)", incident)
    return f"ec-{match.group(1)}" if match else incident.split("_", 1)[0]


def has_polygons(root: str | Path, incident: str) -> bool:
    return (Path(root) / incident / POLYGONS_FILENAME).is_file()


def discover_incidents(root: str | Path) -> list[str]:
    """Every folder under `root` with a `polygons.coco.json`, sorted.

    `polygons.coco.json` is what makes a folder a training example — the rest
    are pulled frames with Labelbox boxes and no corrected masks.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and (d / POLYGONS_FILENAME).is_file())


def auto_train_incidents(root: str | Path, val_incidents: Sequence[str],
                         test_incidents: Sequence[str] | None = None) -> list[str]:
    """Every viable incident under `root` that is not on a held-out camera.

    Executed rather than transcribed, because a literal list goes stale silently
    the moment the labeling queue moves — `auto` resolves to 145 incidents today
    against the 128 `L4E-2` recorded three days ago. The resolved list is written
    into `ckpts/<run>/config_resolved.yaml`, so a finished run still records
    exactly which incidents it saw, and clip count is a resume tripwire for the
    same reason.

    Both held-out sets are excluded **by camera**. `test_incidents` names the GT
    test set, which lives under a different root; naming it here is what makes
    the eval a held-out number rather than one carrying a leakage warning.
    """
    if not val_incidents:
        raise ValueError(
            "splits.train_incidents: auto needs splits.val_incidents. The rule "
            "is 'every viable incident that is not on a held-out camera', so "
            "with no val split there is nothing to hold out and training would "
            "take in the cameras it is about to be scored on.")
    held_out = {camera_of(name) for name in val_incidents}
    held_out |= {camera_of(name) for name in (test_incidents or ())}
    return [name for name in discover_incidents(root)
            if camera_of(name) not in held_out]


def split_viable(root: str | Path, incidents: Iterable[str]) -> tuple[list[str], list[str]]:
    """Partition named incidents into (viable, unlabeled), order preserved.

    A named-but-unlabeled incident is a config that has run ahead of the
    labeling queue, which is routine — it should shrink the set and say so,
    not abort the run.
    """
    viable, unlabeled = [], []
    for incident in incidents:
        (viable if has_polygons(root, incident) else unlabeled).append(incident)
    return viable, unlabeled


def load_polygons(root: str | Path, incident: str
                  ) -> tuple[dict[int, list[list[float]]], tuple[int, int] | None]:
    """`{frame_index: [flat ring, ...]}` and the frame size, from COCO.

    Rings shorter than 3 points are dropped — they cannot be filled and
    `mask_to_polygons` would reject them at the other end anyway. All of a
    frame's annotations are merged into one list: `targets.gt_merge: union` is
    correct here because the multi-instance frames in this corpus are wisps of
    one drifting plume, not separate objects.
    """
    path = Path(root) / incident / POLYGONS_FILENAME
    blob = json.loads(path.read_text(encoding="utf-8"))
    frame_of = {img["id"]: int(img["frame_index"]) for img in blob.get("images", [])}
    image_hw: tuple[int, int] | None = None
    for img in blob.get("images", []):
        if img.get("height") and img.get("width"):
            image_hw = (int(img["height"]), int(img["width"]))
            break
    out: dict[int, list[list[float]]] = {}
    for ann in blob.get("annotations", []):
        frame_index = frame_of.get(ann.get("image_id"))
        if frame_index is None:
            continue
        for ring in ann.get("segmentation") or ():
            if len(ring) < 6:
                continue
            out.setdefault(frame_index, []).append([float(v) for v in ring])
    return out, image_hw


def polygon_frames(root: str | Path, incident: str) -> list[int]:
    """Sorted frame indices that carry a corrected polygon."""
    polygons, _ = load_polygons(root, incident)
    return sorted(polygons)


def frame_path(root: str | Path, incident: str, frame_index: int) -> Path:
    return Path(root) / incident / FRAMES_DIRNAME / f"frame_{frame_index:05d}.jpg"


def smoke_descriptors(root: str | Path, incident: str) -> list[str]:
    """The labeler's free-text notes from `status.json`, or []. 

    Worth reading: this is where "segmented 2x smoke plume well" lives, and it
    is a better multi-plume index than anything derivable from the boxes.
    """
    path = Path(root) / incident / STATUS_FILENAME
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("smoke_descriptors") or [])
    except (OSError, json.JSONDecodeError):
        return []
