"""`training_config.yaml` — the cross-consumer snapshot, and the only code that
reads it.

Four things segment smoke and they have to agree about the same pixels:
training, the eval pipeline, the image repo's `serve.py` and the poc's GUI. The
snapshot holds every value more than one of them needs — the splits, the prompt
geometry, the preprocessing, the masklet linker, the clip corpus, the whole
post-process chain — so the val incident list is written once instead of
duplicated into every config that needs it, which is what went stale the first
time labeling landed.

**The dividing line.** Anything that changes *which samples exist* or *what a
metric means* belongs here. Anything a run varies — lr, epochs, the freeze
schedule, the dropout distribution — lives in `configs/train_video.yaml` and
nowhere else. Every value here is authoritative and something pulls it; if you
want to record something nothing reads, write a comment next to the value.

**`--check` is the contract.** It compares the snapshot against every live
consumer and against the vendored SAM 3 source, and it asserts that each
recorded `divergences:` entry is *still true* — so fixing one without deleting
its entry is a failure, and the list cannot rot into folklore.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = REPO_ROOT / "training_config.yaml"
TRAIN_CONFIG_PATH = REPO_ROOT / "configs" / "train_video.yaml"
VENDORED_GEOMETRY = REPO_ROOT / "vendor" / "smokeseg" / "geometry.py"

# Every top-level section must be in exactly one of these. A section in neither
# is a section nothing reads.
TRAIN_BASE_SECTIONS = {"shared", "splits", "masklets", "clips", "crop",
                       "prompting", "memory", "targets", "loss", "run"}
OTHER_SECTIONS = {"meta", "postprocess", "eval_video", "poc", "poc_field_map",
                  "divergences"}


def load(path: str | Path = SNAPSHOT_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def section(name: str, path: str | Path = SNAPSHOT_PATH) -> dict:
    snap = load(path)
    if name not in snap:
        raise KeyError(f"{name!r} is not a section of {Path(path).name}; "
                       f"sections are {sorted(snap)}")
    return snap[name]


def train_base(snap: dict | None = None) -> dict:
    """The snapshot's schema translated into `Config`'s. The ONLY bridge.

    The snapshot is grouped by consumer; `Config` is grouped by what a run
    varies. Keeping the translation in one function is what lets `--check`
    compare the two without either side guessing.

    Per-experiment knobs are absent by design. A snapshot that supplied
    `train.lr` would be a second place to look for the value a sweep is
    changing.
    """
    snap = snap or load()
    shared, splits = snap["shared"], snap["splits"]
    crop, prompting = snap["crop"], snap["prompting"]
    memory, targets = snap["memory"], snap["targets"]
    return {
        "data": {
            "root": shared["data_root"],
            "train_incidents": splits["train_incidents"],
            "val_incidents": list(splits["val_incidents"]),
            "test_incidents": list(splits["test_incidents"]),
            "gt_dataset_root": splits["gt_dataset_root"],
            "gt_merge": targets["gt_merge"],
            "min_mask_pixels": targets["min_mask_pixels"],
            "clahe": shared["clahe"],
            "clahe_clip": shared["clahe_clip"],
            "clahe_grid": shared["clahe_grid"],
            # Copied one level, not referenced: `_deep_merge` walks into these
            # as a base layer, and a shared dict would let a merge mutate the
            # loaded snapshot.
            "masklets": dict(snap["masklets"]),
            "clips": dict(snap["clips"]),
        },
        "crop": dict(crop),
        "prompt": {
            "box_source": prompting["box_source"],
            "detector_boxes_path": prompting["detector_boxes_path"],
            "mixed_detector_frac": prompting["mixed_detector_frac"],
            # The snapshot records all three carry-forward values because they
            # genuinely differ (divergence `bbox_propagate`); this consumer's
            # is `training`.
            "bbox_propagate_frames": prompting["bbox_propagate_frames"]["training"],
            "bbox_from_polygon": prompting["bbox_from_polygon"],
            "bbox_from_polygon_pad": prompting["bbox_from_polygon_pad"],
            "bbox_event_grouping": dict(prompting["bbox_event_grouping"]),
            # One level deeper: its per-side entries are dicts too.
            "bbox_boundary_extend": {
                k: dict(v) if isinstance(v, dict) else v
                for k, v in prompting["bbox_boundary_extend"].items()},
            "bbox_repair": dict(prompting["bbox_repair"]),
        },
        "model": {
            "sam_version": shared["sam_version"],
            "multimask": shared["multimask"],
            # `train` is absent: which memory slices move is a freeze policy and
            # belongs to the run.
            "memory": {k: v for k, v in memory.items() if k != "train"},
        },
        "train": {"loss": _loss_base(snap["loss"])},
        "run": dict(snap["run"]),
        "feature_cache": {"root": snap["run"]["feature_cache_root"]},
    }


def _loss_base(loss: dict) -> dict:
    return {k: v for k, v in loss.items()}


def flatten(tree: Any, prefix: str = "") -> dict[str, Any]:
    """`{"crop.bbox_pad": 0.5}`. Lists stay whole leaves."""
    out: dict[str, Any] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            out.update(flatten(value, f"{prefix}{key}."))
    else:
        out[prefix.rstrip(".")] = tree
    return out


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_yaml_scalar(v) for v in value) + "]"
    return str(value)


def train_overrides(snap: dict | None = None) -> list[str]:
    """`train_base` rendered as dotted CLI strings, so the two cannot drift."""
    return [f"{k}={_yaml_scalar(v)}" for k, v in flatten(train_base(snap)).items()]


def resolve_path(snap: dict, dotted: str) -> Any:
    node: Any = snap
    for part in dotted.split("."):
        node = node[part]
    return node


def poc_defaults(snap: dict | None = None) -> dict:
    """What the poc layers under its `SMOKESEG_*` env handling."""
    snap = snap or load()
    return {field: resolve_path(snap, dotted)
            for field, dotted in snap["poc_field_map"].items()}


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# --check
# --------------------------------------------------------------------------- #

def check(snap: dict | None = None) -> list[tuple[str, str]]:
    snap = snap or load()
    problems: list[tuple[str, str]] = []
    bad = lambda msg: problems.append(("blocking", msg))

    # --- group 0: every section is read by something --------------------- #
    stray = sorted(set(snap) - TRAIN_BASE_SECTIONS - OTHER_SECTIONS)
    if stray:
        bad(f"training_config.yaml has sections {stray} that nothing reads. "
            "Every value here is supposed to be authoritative and pulled by a "
            "consumer; if you want to record something nothing reads, write a "
            "comment next to the value instead.")

    problems += _check_train_config(snap)
    problems += _check_postprocess(snap)
    problems += _check_geometry(snap)
    problems += _check_launcher(snap)
    problems += _check_filesystem(snap)
    problems += _check_sam3_source(snap)
    problems += _check_divergences(snap)
    return problems


def _check_train_config(snap: dict) -> list[tuple[str, str]]:
    """The resolved run config must equal `train_base`, and must still extend."""
    out: list[tuple[str, str]] = []
    if not TRAIN_CONFIG_PATH.is_file():
        return [("blocking", f"{TRAIN_CONFIG_PATH} does not exist, so nothing "
                             "consumes the snapshot and --check proves nothing.")]
    raw = yaml.safe_load(TRAIN_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if "extends" not in raw:
        out.append(("blocking",
            f"{TRAIN_CONFIG_PATH.name} no longer declares `extends`. Any values "
            "that still agree with the snapshot agree by luck, and luck is not "
            "wiring."))
    # Values stated in both files: the config wins the merge, so the snapshot's
    # copy is a number nobody reads that still looks authoritative.
    overlap = sorted(set(flatten({k: v for k, v in raw.items() if k != "extends"}))
                     & set(flatten(train_base(snap))))
    if overlap:
        out.append(("blocking",
            f"{TRAIN_CONFIG_PATH.name} states {overlap}, which "
            "training_config.yaml also supplies. One value, one place."))
    try:
        from smokeftv.config import load_config
        resolved = load_config(TRAIN_CONFIG_PATH, ["data.train_incidents=[]"])
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        return out + [("blocking",
            f"{TRAIN_CONFIG_PATH.name} does not load: {type(exc).__name__}: {exc}")]
    want, have = flatten(train_base(snap)), flatten(resolved.to_dict())
    for key, value in want.items():
        if key in ("data.train_incidents",):
            continue
        actual = have.get(key, "<absent>")
        if actual != value:
            out.append(("blocking",
                f"{key}: snapshot says {value!r}, the resolved "
                f"{TRAIN_CONFIG_PATH.name} says {actual!r}"))
    return out


def _check_postprocess(snap: dict) -> list[tuple[str, str]]:
    """`postprocess:` against the pinned `PocParams`, both directions."""
    out: list[tuple[str, str]] = []
    try:
        from smokeftv.postprocess import PocParams
    except ImportError:
        return [("blocking", "smokeftv.postprocess does not import, so the "
                             "postprocess: section has no live consumer.")]
    params = PocParams()
    known = {f.name for f in __import__("dataclasses").fields(PocParams)}
    for key, value in snap["postprocess"].items():
        actual = getattr(params, key, "<absent>")
        if actual != value:
            out.append(("blocking",
                f"postprocess.{key}: snapshot says {value!r}, "
                f"surfaces.PocParams says {actual!r}"))
    missing = sorted(known - set(snap["postprocess"]))
    if missing:
        out.append(("blocking",
            f"PocParams pins {missing}, which postprocess: does not state. A "
            "pinned field the snapshot does not name is a metric definition "
            "nobody can see."))
    return out


def _check_geometry(snap: dict) -> list[tuple[str, str]]:
    """The polygon surface IS the poc's `geometry.py`. Hash it.

    The reference repo records `poc_geometry_sha256` and never reads it — a
    manual, unenforced coupling. Here it is load-bearing, because
    `iou_polygon` is only "the surface the GUI ships" if it is this file.
    """
    out: list[tuple[str, str]] = []
    recorded = snap["meta"].get("poc_geometry_sha256")
    if not VENDORED_GEOMETRY.is_file():
        return [("blocking", f"{VENDORED_GEOMETRY} is missing — the polygon "
                             "surface has no implementation to hash.")]
    vendored = _sha16(VENDORED_GEOMETRY)
    if vendored != recorded:
        out.append(("blocking",
            f"vendor/smokeseg/geometry.py hashes to {vendored}, but "
            f"meta.poc_geometry_sha256 says {recorded}. The polygon surface "
            "moved; every recorded iou_polygon was measured under the old one."))
    live = Path(snap["eval_video"]["poc_root"]) / "smokeseg" / "geometry.py"
    if live.is_file():
        live_sha = _sha16(live)
        if live_sha != vendored:
            out.append(("blocking",
                f"the live poc geometry.py ({live}) hashes to {live_sha}, the "
                f"vendored copy to {vendored}. iou_polygon claims to be the "
                "surface the GUI ships while being measured on a different "
                "one. Re-vendor and re-measure, in that order."))
    else:
        out.append(("blocking",
            f"the poc is not mounted at {live}, so the vendored geometry cannot "
            "be checked against the shipping one. Set eval_video.poc_root."))
    return out


def _check_launcher(snap: dict) -> list[tuple[str, str]]:
    """`train.sh` is a second consumer of `run:` and cannot import anything."""
    out: list[tuple[str, str]] = []
    launcher = REPO_ROOT / "train.sh"
    if not launcher.is_file():
        return [("blocking", "train.sh is missing.")]
    text = launcher.read_text(encoding="utf-8")
    run = snap["run"]
    for needle, key, value in (
        (f'CKPT_DIR="{run["ckpts_root"]}/${{RUN_NAME}}"', "run.ckpts_root", run["ckpts_root"]),
        (f'LOG_PATH="{run["logs_root"]}/${{RUN_NAME}}.log"', "run.logs_root", run["logs_root"]),
    ):
        if needle not in text:
            out.append(("blocking",
                f"train.sh does not contain {needle!r}, so {key}: {value!r} "
                "describes a layout the launcher does not build."))
    if run["run_name_from"] == "stdout" and "readlink -f /proc/self/fd/1" not in text:
        out.append(("blocking",
            "run.run_name_from is 'stdout' but train.sh does not resolve its "
            "own stdout, so ckpts/<run>/ and logs/<run>.log can drift apart."))
    if run["require_clean_git"] and "--allow-dirty" not in text:
        out.append(("blocking",
            "run.require_clean_git is true but train.sh has no dirty-tree gate."))
    return out


def _check_filesystem(snap: dict) -> list[tuple[str, str]]:
    """The failures that cost forty minutes instead of a second."""
    out: list[tuple[str, str]] = []
    for key, path in (("meta.init_from", snap["meta"]["init_from"]),
                      ("shared.data_root", snap["shared"]["data_root"]),
                      ("splits.gt_dataset_root", snap["splits"]["gt_dataset_root"])):
        if not Path(path).exists():
            out.append(("blocking", f"{key}: {path} does not exist."))
    arm = Path(snap["eval_video"]["baseline_arm"])
    for name in ("best.pt", "config_resolved.yaml"):
        if not (arm / name).is_file():
            out.append(("blocking",
                f"eval_video.baseline_arm has no {name} at {arm}. Arm 1 is the "
                "number every video arm is compared against."))

    # The GT set's cameras must be a subset of the held-out cameras, or a run
    # trains on a camera it is later reported against.
    from smokeftv.incidents import camera_of, discover_incidents
    gt_root = Path(snap["splits"]["gt_dataset_root"])
    if gt_root.is_dir():
        try:
            gt_cams = {camera_of(n) for n in discover_incidents(gt_root)}
        except FileNotFoundError:
            gt_cams = set()
        held = {camera_of(n) for n in snap["splits"]["val_incidents"]}
        held |= {camera_of(n) for n in snap["splits"]["test_incidents"]}
        leaked = sorted(gt_cams - held)
        if leaked:
            out.append(("blocking",
                f"cameras {leaked} are scored under splits.gt_dataset_root but "
                "are not held out by splits.val_incidents / test_incidents, so "
                "`auto` will train on them the moment one is labelled. Name an "
                "incident on each in test_incidents."))

    # Disk. The root filesystem is full on this box; a run that dies writing
    # last_resume.pt after an hour of training is the failure this prevents.
    for key, path, need_gb in (
        ("run.feature_cache_root", snap["run"]["feature_cache_root"], 100),
        ("run.ckpts_root", str(REPO_ROOT / snap["run"]["ckpts_root"]), 20),
        ("run.logs_root", str(REPO_ROOT / snap["run"]["logs_root"]), 1),
    ):
        probe = Path(path)
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            free_gb = shutil.disk_usage(probe).free / 1e9
        except OSError:
            continue
        if free_gb < need_gb:
            out.append(("blocking",
                f"{key} -> {path} has {free_gb:.0f} GB free, under the {need_gb} "
                f"GB this needs. The feature cache alone is ~57 GiB."))
    return out


def _check_sam3_source(snap: dict) -> list[tuple[str, str]]:
    """Assertions about the vendored tracker that the snapshot depends on."""
    out: list[tuple[str, str]] = []
    from smokeftv.config import SAM3_REPO_ROOT
    tracker = SAM3_REPO_ROOT / "sam3" / "model" / "sam3_tracker_base.py"
    predictor = SAM3_REPO_ROOT / "sam3" / "model" / "sam3_tracking_predictor.py"
    if not tracker.is_file() or not predictor.is_file():
        return [("blocking", f"the sam3 submodule is not checked out at "
                             f"{SAM3_REPO_ROOT}.")]
    src = tracker.read_text(encoding="utf-8")
    psrc = predictor.read_text(encoding="utf-8")
    mem = snap["memory"]

    if "self.add_all_frames_to_correct_as_cond" not in psrc:
        out.append(("blocking",
            "Sam3TrackerPredictor no longer sets "
            "add_all_frames_to_correct_as_cond, which divergence "
            "`memory_conditioning` is about. Re-read the vendored source."))
    # memory.temporal_pos_encoding / object_pointers are assertions, not knobs.
    if mem["temporal_pos_encoding"] and "maskmem_tpos_enc" not in src:
        out.append(("blocking",
            "memory.temporal_pos_encoding asserts maskmem_tpos_enc exists on "
            "Sam3TrackerBase; it does not."))
    if mem["object_pointers"] and "obj_ptr_proj" not in src:
        out.append(("blocking",
            "memory.object_pointers asserts obj_ptr_proj exists on "
            "Sam3TrackerBase; it does not."))
    if "torch.autocast" not in psrc:
        out.append(("blocking",
            "Sam3TrackerPredictor no longer enters a process-global autocast in "
            "__init__. model.build_model exits it explicitly; if upstream "
            "stopped doing it, that call is now a no-op that will read as one."))
    return out


def _check_divergences(snap: dict) -> list[tuple[str, str]]:
    """Each recorded divergence must still be TRUE.

    The inversion is the point. A divergence that stops being true is a
    failure, so the list cannot rot into folklore describing a disagreement
    somebody already fixed.
    """
    out: list[tuple[str, str]] = []
    bad = lambda msg: out.append(("blocking", msg))
    ids = {d["id"] for d in snap.get("divergences", [])}
    crop, clips = snap["crop"], snap["clips"]
    prompting, ev = snap["prompting"], snap["eval_video"]
    loss, masklets, memory = snap["loss"], snap["masklets"], snap["memory"]

    if "crop_mode" in ids and crop["mode"] != "track_window":
        bad(f"divergence 'crop_mode' describes video training on a fixed "
            f"track_window against the image repo's per-frame compute_window "
            f"and the poc's compute_crop — three windows. crop.mode is "
            f"{crop['mode']!r}: at 'per_frame_box' the video arm has adopted "
            "the image window and there are two; at 'full_frame' there is none. "
            "Rewrite the entry or delete it.")

    if "windowing_inherited" in ids:
        if crop.get("bbox_pad") is None:
            bad("divergence 'windowing_inherited' claims per_frame_box uses "
                "compute_window while the poc uses compute_crop, but "
                "crop.bbox_pad is null — crops.crop_for_box branches on exactly "
                "that, so both are now on compute_crop. Converged; delete it.")
        if not {"crop_pad_frac", "crop_pad_top_extra"} <= set(crop):
            bad("divergence 'windowing_inherited' says poc_drift compares "
                "crop_pad_frac / crop_pad_top_extra on an axis the finetune no "
                "longer reads. Those keys are gone from crop:, so the entry has "
                "nothing left to point at.")

    gaps = prompting["bbox_propagate_frames"]
    if "bbox_propagate" in ids and len({gaps["training"], gaps["poc"],
                                        gaps["dataclass_default"]}) == 1:
        bad(f"divergence 'bbox_propagate' records three values for how far a "
            f"prompt box carries forward, but all three are now "
            f"{gaps['training']}. Delete the entry so --check starts enforcing "
            "agreement — and note it decides clip count, which is a resume "
            "tripwire.")

    if "frame_cap" in ids:
        if clips["frame_cap"] != ev["frame_cap"]:
            bad(f"divergence 'frame_cap' claims clips.frame_cap matches the "
                f"eval's so training and scoring share a horizon, but they are "
                f"{clips['frame_cap']} and {ev['frame_cap']}. That is a "
                "different and worse divergence than the one recorded.")
        if not clips["frame_cap"]:
            bad("divergence 'frame_cap' describes a 60-frame horizon the poc "
                "does not have; the cap is off, so the divergence is resolved.")

    if "no_two_plume" in ids:
        if clips["max_masklets_per_clip"] != 1:
            bad(f"divergence 'no_two_plume' records that two-plume work is out "
                f"of scope, but clips.max_masklets_per_clip is "
                f"{clips['max_masklets_per_clip']}. loss.exclusion_weight, the "
                "two oversamplers and the id_swap_rate / idf1 / fragmentation "
                "track metrics all have to come back with it.")
        readded = [f"clips.{k}" for k in ("multi_plume_oversample",
                                          "merge_frame_oversample") if k in clips]
        readded += [f"loss.{k}" for k in ("exclusion_weight",) if k in loss]
        readded += [f"eval_video.track_metrics:{m}"
                    for m in ("id_swap_rate", "idf1", "fragmentation")
                    if m in ev["track_metrics"]]
        if readded:
            bad(f"divergence 'no_two_plume' says these were removed, but the "
                f"snapshot states {sorted(readded)}. Config for a feature the "
                "code does not have is exactly what unknown-keys-raise exists "
                "to prevent.")

    if "detector_prompts" in ids:
        if prompting["box_source"] != "annotations":
            bad(f"divergence 'detector_prompts' records that only 'annotations' "
                f"is implemented, but prompting.box_source is "
                f"{prompting['box_source']!r}.")
        if prompting["detector_boxes_path"] is not None:
            bad("divergence 'detector_prompts' says there is no detector dump, "
                f"but prompting.detector_boxes_path is "
                f"{prompting['detector_boxes_path']!r}. A path nothing reads is "
                "worse than null.")

    if "no_smokebase" in ids and (loss["smokebase_weight"] != 0.0
                                  or masklets["smokebase_crosscheck"]):
        bad(f"divergence 'no_smokebase' records that cvf-2026 carries no "
            f"hand-placed smoke-base labels, but loss.smokebase_weight is "
            f"{loss['smokebase_weight']} and masklets.smokebase_crosscheck is "
            f"{masklets['smokebase_crosscheck']}. Both read labels that are not "
            "there. smokebase_recall stays a metric either way.")

    if "bbox_repair_for_baseline" in ids:
        if prompting["bbox_repair"]["enabled"]:
            bad("divergence 'bbox_repair_for_baseline' records that bbox_repair "
                "is OFF for training because it reads the GT to rewrite the "
                "prompt. It is enabled, so training is learning a prompt no "
                "detector produces.")
        if abs(float(ev["baseline_iou_polygon"]) - 0.6620960408) > 1e-9:
            bad("divergence 'bbox_repair_for_baseline' is about reproducing "
                f"0.6620960408; eval_video.baseline_iou_polygon is "
                f"{ev['baseline_iou_polygon']}. Requoting the target without "
                "rewriting the entry loses the reason the number has two values.")

    if "fuse_fill_frac" in ids:
        try:
            from smokeftv.metrics import all_ious, fuse_soft
        except ImportError:
            pass
        else:
            if "max_fill_frac" in inspect.getsource(all_ious):
                bad("divergence 'fuse_fill_frac' records that metrics.all_ious "
                    "calls fuse_soft WITHOUT max_fill_frac. It now passes it — "
                    "which is the right fix, and it moves iou_fused off the "
                    "recorded 0.6740399427. Re-measure, then delete the entry.")
            default = inspect.signature(fuse_soft).parameters["max_fill_frac"].default
            if snap["shared"]["max_fill_frac"] != default:
                bad(f"shared.max_fill_frac is {snap['shared']['max_fill_frac']} "
                    f"but metrics.fuse_soft defaults to {default}, so the two "
                    "surfaces no longer filter the same candidates.")

    if "memory_conditioning" in ids and memory["prompted_frames_are_conditioning"]:
        bad("divergence 'memory_conditioning' records that this repo sets "
            "prompted_frames_are_conditioning false against upstream's "
            "hardcoded true. It is now true, so per-frame prompting makes EVERY "
            "frame conditioning: the cond cap saturates and the recency FIFO is "
            "evicted.")

    if "test_split_is_not_the_gt_set" in ids:
        from smokeftv.incidents import discover_incidents
        gt_root = Path(snap["splits"]["gt_dataset_root"])
        if gt_root.is_dir():
            gt = set(discover_incidents(gt_root))
            if gt == set(snap["splits"]["test_incidents"]):
                bad("divergence 'test_split_is_not_the_gt_set' records that the "
                    "two lists are different things; they are now identical. "
                    "Delete the entry, or restore the camera-exclusion names "
                    "that are not scored.")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m smokeftv.config_all",
        description=__doc__.split("\n")[0])
    parser.add_argument("-f", "--file", default=SNAPSHOT_PATH, type=Path)
    parser.add_argument("--pedantic", action="store_true",
                        help="advisories fail too")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--show", action="store_true")
    group.add_argument("--train-overrides", action="store_true")
    group.add_argument("--poc-defaults", action="store_true")
    group.add_argument("--data-root", action="store_true")
    args = parser.parse_args(argv)

    snap = load(args.file)
    if args.data_root:
        print(snap["shared"]["data_root"])
        return 0
    if args.show:
        print(yaml.safe_dump(snap, sort_keys=False))
        return 0
    if args.train_overrides:
        print("\n".join(train_overrides(snap)))
        return 0
    if args.poc_defaults:
        for key, value in sorted(poc_defaults(snap).items()):
            print(f"{key}={_yaml_scalar(value)}")
        return 0

    problems = check(snap)
    advisories = [m for sev, m in problems if sev == "advisory"]
    blocking = [m for sev, m in problems if sev == "blocking"]
    # Advisories go to stdout deliberately: under `set -e` in a preflight they
    # must not read as the failure.
    for message in advisories:
        print(f"  advisory: {message}")
    if not blocking:
        n_div = len(snap.get("divergences", []))
        print(f"{args.file.name} agrees with every consumer "
              f"(snapshot {snap['meta']['snapshot_date']}, "
              f"{n_div} divergence(s) still true)")
        return 1 if (advisories and args.pedantic) else 0

    print(f"{args.file.name} disagrees with reality in {len(blocking)} place(s) "
          "that a consumer actually pulls:", file=sys.stderr)
    for message in blocking:
        print(f"  - {message}", file=sys.stderr)
    print("\nEach line above is a value two programs read differently, or a "
          "recorded divergence that is no longer true. Fix the consumer, or "
          "rewrite the divergence. Do not 'fix' this by changing the snapshot "
          "to match a value you did not intend to change.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
