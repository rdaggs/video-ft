#!/usr/bin/env python
"""A run's checkpoint against the weights it started from, on the GT test set.

    CUDA_VISIBLE_DEVICES=6 uv run python scripts/eval_video.py --run ckpts/<run>

The video counterpart of `sam3-finetuned-cvf/eval_pipeline/gt_test_set.py`. A
run's `log.csv` is about its val cameras; "how does this checkpoint land on the
GT test set" is a different question against a different dataset
(`splits.gt_dataset_root`, 28 incidents), and it needs the checkpoint run over
that set.

Two arms, always scored under one measurement:

  init        the weights the run STARTED from — released SAM 3 plus the
              `init_from` overlay (L4E-2). The memory path is stock, so this is
              the video arm's zero-shot bar, and the delta against it is what
              the memory training bought.
  finetuned   init + `best.pt` (or `--ckpt`, e.g. `last.pt`, named `last`).

Rows are cached per arm under a key covering everything the numbers depend on
except the weights, so every run sharing a measurement reads one `init` table:

    <eval_cache>/gt_test_set/<key>/init-<sha16>/frames.csv
    <eval_cache>/gt_test_set/<key>/ckpt-<sha16>/frames.csv
    ckpts/<run>/gt_test_set/{manifest.json,report.txt,<arm>/...}   <- the run's copy

The run's copy is complete on its own — both arms' rows, not pointers — because
`finalize_run` commits it, and a record that points into a cache is not one.

The eval settings come from the run's own `config_resolved.yaml` with the data
root pointed at the GT set; see `smokeftv.evaluate` for what else is pinned.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smokeftv import config_all, runlog                         # noqa: E402
from smokeftv.config import load_config                         # noqa: E402
from smokeftv.incidents import camera_of                        # noqa: E402

SCHEMA = 1
INIT_ARM = "init"
# The surface every per-t / per-kind / per-incident / worst-frame table ranks
# on. "fused" matches what train.py logs and selects best.pt on; "polygon" is
# the GUI's full-frame surface. Both stay in frames.csv and the headline.
SURFACE = "fused"
T_METRIC = f"iou_{SURFACE}"
HEADLINE_MODE = "conditional"
KINDS = ("cond", "prompted", "propagated")

FRAME_FIELDS = [
    "sample_id", "mode", "incident", "camera", "masklet_id", "clip_id", "t",
    "frame_index", "stride", "kind", "prompted", "reprompted", "box_gap",
    "obj_score", "iou_fused", "iou_selected", "iou_best", "prec_fused",
    "recall_fused", "gt_px_fused", "pred_px_fused", "iou_polygon",
    "prec_polygon", "recall_polygon", "gt_px_polygon", "pred_px_polygon",
    "gt_px_outside_crop", "continuity", "gt_continuity",
]
MEAN_FIELDS = ["iou_polygon", "iou_fused", "iou_selected", "iou_best",
               "prec_polygon", "recall_polygon", "prec_fused", "recall_fused"]
SCORER_FILES = ["smokeftv/evaluate.py", "smokeftv/rollout.py",
                "smokeftv/track_step.py", "smokeftv/postprocess.py",
                "smokeftv/metrics.py", "smokeftv/dataset.py", "smokeftv/corpus.py"]


# --------------------------------------------------------------------------- #
# Paths, hashes, tables — torch-free
# --------------------------------------------------------------------------- #

def resolve(path: str | Path) -> Path:
    out = Path(path)
    return out if out.is_absolute() else REPO_ROOT / out


def sha16(path: str | Path) -> str:
    return runlog.sha256_file(resolve(path), short=True)


def eval_cache_root(cfg) -> Path:
    # Beside the feature cache on /data3: the root filesystem is full.
    return Path(cfg.run.feature_cache_root).parent / "eval_cache" / "gt_test_set"


def read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["mode"], r["incident"],
                                               r["clip_id"], int(r["t"]))):
            writer.writerow(row)
    tmp.replace(path)


def _floats(rows, field) -> list[float]:
    return [float(r[field]) for r in rows if r.get(field) not in (None, "")]


def _mean(rows, field) -> float | None:
    values = _floats(rows, field)
    return statistics.fmean(values) if values else None


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    line = lambda cells: "| " + " | ".join(
        str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([rule, line(headers), rule, *map(line, rows), rule])


def fmt(value, spec=".4f") -> str:
    return "—" if value is None else format(value, spec)


def px(value) -> str:
    return f"{int(float(value)):,}"


# --------------------------------------------------------------------------- #
# Summaries — what metrics.jsonl carries for training, for the test set
# --------------------------------------------------------------------------- #

def summarize(rows: list[dict], modes) -> dict:
    """Per mode: means, iou_by_t, iou_by_kind, and the track metrics."""
    out = {}
    for mode in modes:
        group = [r for r in rows if r["mode"] == mode]
        if not group:
            continue
        length = max(int(r["t"]) for r in group) + 1
        by_t = {m: [_mean([r for r in group if int(r["t"]) == t], m)
                    for t in range(length)] for m in ("iou_polygon", "iou_fused")}
        clips = {r["clip_id"] for r in group}
        positive = [r for r in group if float(r["gt_px_fused"]) > 0]
        out[mode] = {
            "samples": len(group),
            "clips": len(clips),
            "incidents": len({r["incident"] for r in group}),
            **{m: _mean(group, m) for m in MEAN_FIELDS},
            "iou_by_t": by_t,
            "iou_by_kind": {m: {k: _mean([r for r in group if r["kind"] == k], m)
                                for k in KINDS}
                            for m in ("iou_polygon", "iou_fused")},
            "continuity": _mean(group, "continuity"),
            "gt_continuity": _mean(group, "gt_continuity"),
            # The presence head said "no object" on a frame that has one: the
            # mask is replaced by a constant and the frame's memory is poisoned.
            "lost_frac": (sum(1 for r in positive if float(r["obj_score"]) <= 0)
                          / len(positive)) if positive else None,
            "restarts_per_clip": sum(int(r["reprompted"]) for r in group) / len(clips),
        }
    return out


def _worst_key(row: dict) -> tuple[float, float]:
    """Lowest IoU first, ties to the biggest GT plume."""
    return float(row[T_METRIC]), -float(row[f"gt_px_{SURFACE}"])


def incident_rows(rows: list[dict], modes) -> list[dict]:
    out = []
    for mode in modes:
        by_incident: dict[str, list[dict]] = {}
        for row in rows:
            if row["mode"] == mode:
                by_incident.setdefault(row["incident"], []).append(row)
        for incident, group in sorted(by_incident.items()):
            worst = min(group, key=_worst_key)
            record = {"mode": mode, "incident": incident,
                      "camera": camera_of(incident), "n": len(group),
                      "clips": len({r["clip_id"] for r in group})}
            for field in MEAN_FIELDS:
                record[field] = round(_mean(group, field), 6)
            record[f"median_{T_METRIC}"] = round(statistics.median(
                _floats(group, T_METRIC)), 6)
            record["worst_frame"] = int(worst["frame_index"])
            record["worst_t"] = int(worst["t"])
            record[f"worst_{T_METRIC}"] = round(float(worst[T_METRIC]), 6)
            out.append(record)
    return out


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

def render_report(manifest: dict, arm_rows: dict[str, list[dict]]) -> str:
    modes = manifest["modes"]
    arms = [a["name"] for a in manifest["arms"]]
    base, arm = arms[0], arms[-1]
    summaries = {name: summarize(rows, modes) for name, rows in arm_rows.items()}
    length = max(len(s["iou_by_t"]["iou_polygon"])
                 for per in summaries.values() for s in per.values())
    lines = [
        f"run_name: {manifest['run']}",
        f"GT test set {manifest['data_root']}",
        f"  {manifest['num_incidents']} incidents, {manifest['num_clips']} clips, "
        f"{manifest['num_clips'] * length} (clip, t) samples per mode   "
        f"cache_key {manifest['cache_key']}",
        f"  {arm}: {manifest['arms'][-1]['ckpt']}",
        f"  held out: {len(manifest['trained_incidents'])} trained incidents, "
        f"{len(manifest['trained_cameras'])} shared cameras",
    ]
    if manifest.get("box_jitter"):
        low, high = manifest["box_jitter"]
        lines.append(f"  BOX JITTER: every prompt box padded per axis by p ~ "
                     f"U({low}, {high}), seeded per frame — the same boxes for every run")
    if manifest.get("sample_limit"):
        lines.append(f"  --limit {manifest['sample_limit']} clips: a smoke test, "
                     "NOT a result")
    for warning in manifest.get("warnings", []):
        lines.append(f"  WARNING {warning}")

    head = []
    for name in arms:
        for mode in modes:
            s = summaries[name].get(mode)
            if s is None:
                continue
            b = summaries[base].get(mode, {})
            delta = {m: None if name == base or b.get(m) is None else s[m] - b[m]
                     for m in ("iou_polygon", "iou_fused")}
            head.append([name + (" (baseline)" if name == base else ""), mode,
                         fmt(s["iou_fused"]), fmt(delta["iou_fused"], "+.4f"),
                         fmt(s["prec_fused"]), fmt(s["recall_fused"]),
                         fmt(s["iou_polygon"]), fmt(delta["iou_polygon"], "+.4f"),
                         fmt(s["iou_best"])])
    lines += ["", "HEADLINE  (mean over (clip, t) samples; best-of-3 is the "
                  "GT-picked candidate, an oracle)",
              render_table(["arm", "mode", "IoU fused", "delta", "prec fused",
                            "recall fused", "IoU polygon", "delta", "best-of-3"],
                           head)]

    by_t = []
    for name in arms:
        for mode in modes:
            s = summaries[name].get(mode)
            if s is None:
                continue
            values = s["iou_by_t"][T_METRIC]
            tail = [v for v in values[1:] if v is not None]
            by_t.append([name, mode, *[fmt(v, ".3f") for v in values],
                         fmt(statistics.fmean(tail) if tail else None, ".3f"),
                         fmt(values[-1] - values[1] if len(values) > 1
                             and None not in (values[1], values[-1]) else None,
                             "+.3f")])
    lines += ["", f"{T_METRIC} BY CLIP POSITION t  (a memory bug is a monotone "
                  "decay in t)",
              render_table(["arm", "mode", *[f"t{t}" for t in range(length)],
                            "t1-7 mean", f"t{length - 1}-t1"], by_t)]

    kinds, video = [], []
    for name in arms:
        for mode in modes:
            s = summaries[name].get(mode)
            if s is None:
                continue
            kinds.append([name, mode, *[fmt(s["iou_by_kind"][T_METRIC][k], ".3f")
                                        for k in KINDS]])
            video.append([name, mode, fmt(s["continuity"], ".3f"),
                          fmt(s["gt_continuity"], ".3f"),
                          fmt(s["lost_frac"], ".3f"),
                          fmt(s["restarts_per_clip"], ".2f")])
    lines += ["", f"{T_METRIC} BY FRAME KIND",
              render_table(["arm", "mode", *KINDS], kinds),
              "", "TRACK METRICS  (continuity = IoU of consecutive predicted "
                  "masks; lost = presence head <= 0 on a GT-positive frame)",
              render_table(["arm", "mode", "continuity", "GT continuity",
                            "lost frac", "restarts/clip"], video)]

    mode = HEADLINE_MODE if HEADLINE_MODE in modes else modes[0]
    base_inc = {r["incident"]: r for r in incident_rows(arm_rows[base], [mode])}
    arm_inc = {r["incident"]: r for r in incident_rows(arm_rows[arm], [mode])}
    paired = {r["sample_id"]: float(r[T_METRIC])
              for r in arm_rows[base] if r["mode"] == mode}
    per_inc = []
    for incident in sorted(base_inc, key=lambda i: base_inc[i][T_METRIC]):
        if incident not in arm_inc:
            continue
        rows = [r for r in arm_rows[arm]
                if r["mode"] == mode and r["incident"] == incident]
        up = sum(1 for r in rows if float(r[T_METRIC]) > paired.get(r["sample_id"], 1e9))
        per_inc.append([incident, str(len(rows)), f"{base_inc[incident][T_METRIC]:.4f}",
                        f"{arm_inc[incident][T_METRIC]:.4f}",
                        f"{arm_inc[incident][T_METRIC] - base_inc[incident][T_METRIC]:+.4f}",
                        f"{up}/{len(rows)}"])
    lines += ["", f"PER INCIDENT  ({mode}, {T_METRIC}, worst baseline first)",
              render_table(["incident", "n", base, arm, "delta", "samples up"],
                           per_inc)]

    worst = []
    rows_by_incident: dict[str, list[dict]] = {}
    for row in arm_rows[arm]:
        if row["mode"] == mode:
            rows_by_incident.setdefault(row["incident"], []).append(row)
    for incident, group in rows_by_incident.items():
        worst.append(min(group, key=_worst_key))
    worst.sort(key=_worst_key)
    lines += ["", f"WORST SINGLE FRAME IN EACH  ({arm}, {mode}, {T_METRIC}; lowest "
                  "IoU, ties to the biggest GT plume)",
              render_table(["incident", "t", "frame", "IoU", "prec", "recall",
                            "GT px", "SAM 3 px"],
                           [[r["incident"], r["t"], r["frame_index"],
                             f"{float(r[T_METRIC]):.3f}",
                             f"{float(r[f'prec_{SURFACE}']):.3f}",
                             f"{float(r[f'recall_{SURFACE}']):.3f}",
                             px(r[f"gt_px_{SURFACE}"]), px(r[f"pred_px_{SURFACE}"])]
                            for r in worst])]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Building it — the only part that needs a model
# --------------------------------------------------------------------------- #

def cache_key(cfg, params, poc, geometry_sha: str, overlays_config: dict) -> str:
    """Everything the rows depend on except the weights and the incident list."""
    prompt = asdict(cfg.prompt)
    # Absent unless on, so keys scored before the field existed stay valid.
    if not cfg.prompt.box_jitter.enabled:
        prompt.pop("box_jitter")
    payload = {
        "root": cfg.data.root,
        "clips": asdict(cfg.data.clips),
        "masklets": asdict(cfg.data.masklets),
        "crop": asdict(cfg.crop),
        "prompt": prompt,
        "clahe": [cfg.data.clahe, cfg.data.clahe_clip, cfg.data.clahe_grid],
        "gt": [cfg.data.gt_merge, cfg.data.min_mask_pixels],
        "model": [cfg.model.sam_version, cfg.model.multimask_mode,
                  cfg.model.mem_mask_source, cfg.model.memory.max_recent_frames,
                  cfg.model.memory.max_cond_frames_in_attn,
                  cfg.model.memory.keep_first_cond_frame],
        "resolution": [cfg.crop.image_size, cfg.train.loss_size],
        "amp": cfg.train.amp_dtype,
        "eval": asdict(params),
        "poc": asdict(poc),
        "geometry": geometry_sha,
        "scorer": {f: sha16(f) for f in SCORER_FILES},
        **overlays_config,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def init_overlays(run_dir: Path, cfg) -> tuple[list[str], list[str]]:
    """The overlays the run started from, and any warning about them."""
    warnings = []
    prov = read_json(run_dir / "provenance.json")
    recorded = (prov.get("init_from") or {})
    if cfg.train.init_ckpt:
        paths = ([cfg.train.init_ckpt] if isinstance(cfg.train.init_ckpt, str)
                 else list(cfg.train.init_ckpt))
    elif recorded.get("path"):
        paths = [recorded["path"]]
        if recorded.get("sha256") and Path(paths[0]).is_file() and \
                runlog.sha256_file(paths[0]) != recorded["sha256"]:
            warnings.append(f"{paths[0]} changed since the run recorded its hash")
    else:
        paths = [config_all.load()["meta"]["init_from"]]
        warnings.append("provenance.json records no init_from; using the live "
                        "snapshot's meta.init_from")
    for path in paths:
        if not Path(path).is_file():
            raise SystemExit(f"init overlay {path} does not exist")
    return paths, warnings


def pick_ckpt(run_dir: Path, request: str | None, name: str | None) -> tuple[str, Path]:
    if request:
        path = resolve(request)
        return name or {"best": "finetuned"}.get(path.stem, path.stem), path
    for arm, filename in (("finetuned", "best.pt"), ("last", "last.pt")):
        if (run_dir / filename).is_file():
            return name or arm, run_dir / filename
    raise SystemExit(f"{run_dir} has neither best.pt nor last.pt, so there is no "
                     "checkpoint arm to score. Pass --ckpt.")


def covers(rows: list[dict], expected: set[str], modes) -> str | None:
    if not rows:
        return "the cached table is empty"
    missing = [f for f in FRAME_FIELDS if f not in rows[0]]
    if missing:
        return f"the cached table predates column(s) {missing}"
    have = {(r["mode"], r["sample_id"]) for r in rows}
    want = {(m, s) for m in modes for s in expected}
    if want - have:
        return f"missing {len(want - have)} of {len(want)} samples"
    return None


def out_dir_name(box_jitter: tuple[float, float] | None) -> str:
    return "gt_test_set" if box_jitter is None else "gt_test_set_boxjitter"


def build(run_dir: Path, *, ckpt_request: str | None, arm_name: str | None,
          modes: tuple[str, ...] | None, limit: int | None, refresh: bool,
          box_jitter: tuple[float, float] | None = None) -> int:
    run_dir = resolve(run_dir)
    config = run_dir / "config_resolved.yaml"
    if not config.is_file():
        raise SystemExit(f"{config} is missing. Without it the geometry this "
                         "checkpoint was trained under is unknown.")
    out_dir = run_dir / out_dir_name(box_jitter)
    log = setup_logging(out_dir)

    import torch

    from smokeftv import checkpoint as ckpt_io
    from smokeftv.evaluate import (MODES, EvalParams, eval_clips, eval_overrides,
                                   sample_id, score_clips)
    from smokeftv.model import build_tracker, prepare_tracker_for_training
    from smokeftv.postprocess import PocParams, VENDORED_GEOMETRY, vendored_geometry

    probe = load_config(config)
    gt_root = probe.data.gt_dataset_root
    cfg = load_config(config, [*eval_overrides(gt_root, box_jitter),
                               f"run_name=gt_test_set_{run_dir.name}"])
    ev = config_all.load()["eval_video"]
    params = EvalParams(modes=tuple(modes or ev.get("inference_modes") or MODES),
                        keyframe_every=int(ev.get("keyframe_every", 8)),
                        conditional_reprompt_iou=float(ev.get("conditional_reprompt_iou", 0.5)))
    poc = PocParams()
    geometry = vendored_geometry()

    name, ckpt_path = pick_ckpt(run_dir, ckpt_request, arm_name)
    if not ckpt_path.is_file():
        raise SystemExit(f"{ckpt_path} does not exist")
    init_paths, warnings = init_overlays(run_dir, cfg)
    key = cache_key(cfg, params, poc, runlog.sha256_file(VENDORED_GEOMETRY, short=True),
                    {"base": "released sam3"})

    incidents, clips, reports = eval_clips(cfg)
    if limit:
        clips = clips[:limit]
    expected = {sample_id(c.clip_id, t) for c in clips
                for t in range(len(c.frame_indices))}
    trained = set(cfg.data.train_incidents)
    trained_incidents = sorted(trained & set(incidents))
    trained_cameras = sorted({camera_of(n) for n in trained}
                             & {camera_of(n) for n in incidents})
    log.info("GT test set for %s", run_dir.name)
    log.info("  %s: %d incidents, %d clips, modes %s", gt_root, len(incidents),
             len(clips), ", ".join(params.modes))
    log.info("  cache_key %s", key)
    for report in reports:
        if not report.clips:
            log.info("  no clips: %s", report)

    init_slug = "init-" + "-".join(sha16(p) for p in init_paths)
    arms = [(INIT_ARM, init_paths, init_slug),
            (name, [*init_paths, str(ckpt_path)], "ckpt-" + sha16(ckpt_path))]

    tracker = None
    loaded: list[str] = []
    entries, arm_rows = [], {}
    for arm, overlays, slug in arms:
        directory = eval_cache_root(cfg) / key / slug
        rows_path = directory / "frames.csv"
        gap = "--refresh" if refresh else (
            covers(read_rows(rows_path), expected, params.modes)
            if rows_path.is_file() else "not scored yet")
        if gap is None:
            log.info("  %s: cached at %s", arm, directory)
            rows = [r for r in read_rows(rows_path)
                    if r["sample_id"] in expected and r["mode"] in params.modes]
        else:
            log.info("  %s: scoring (%s)", arm, gap)
            if tracker is None:
                tracker = build_tracker(cfg.device)
            # Overlays are disjoint by construction (L4E-2 owns the trunk, a
            # memory checkpoint owns the tracker), so the checkpoint arm is the
            # init arm plus one more overlay rather than a second 3.4 GB build.
            new = [p for p in overlays if p not in loaded]
            if loaded and loaded != overlays[:len(loaded)]:
                tracker, loaded, new = build_tracker(cfg.device), [], list(overlays)
            if new:
                ckpt_io.load_overlays(new, tracker)
                loaded += new
            prepare_tracker_for_training(tracker, cfg, seed=cfg.seed)
            tracker.requires_grad_(False)
            tracker.eval()
            rows = score_clips(tracker, clips, cfg, params=params, poc=poc,
                               geometry=geometry)
            if not limit:
                write_rows(rows_path, rows, FRAME_FIELDS)
                (directory / "provenance.json").write_text(json.dumps({
                    "cache_key": key, "arm": arm, "overlays": overlays,
                    "overlay_sha256": [runlog.sha256_file(p) for p in overlays],
                    "config_source": str(config), "data_root": gt_root,
                    "eval": asdict(params), "scored_utc": runlog.utcnow(),
                    "git": runlog.git_info(REPO_ROOT),
                    "env": runlog.env_fingerprint(),
                }, indent=2), encoding="utf-8")
            torch.cuda.empty_cache()
        arm_rows[arm] = rows
        entries.append({"name": arm, "ckpt": overlays[-1] if arm != INIT_ARM else None,
                        "overlays": overlays,
                        "overlay_sha256": [runlog.sha256_file(p) for p in overlays],
                        "cache_dir": str(directory)})
        summary = summarize(rows, params.modes)
        for mode, s in summary.items():
            log.info("  %-9s %-18s iou_fused %.4f  iou_polygon %.4f  by_t %s",
                     arm, mode, s["iou_fused"], s["iou_polygon"],
                     [None if v is None else round(v, 3)
                      for v in s["iou_by_t"][T_METRIC]])

    manifest = {
        "schema": SCHEMA,
        "run": run_dir.name,
        "cache_key": key,
        "data_root": gt_root,
        "box_jitter": list(box_jitter) if box_jitter else None,
        "config_source": str(config),
        "config_sha256": runlog.sha256_file(config),
        "modes": list(params.modes),
        "eval": asdict(params),
        "incidents": incidents,
        "num_incidents": len({c.incident for c in clips}),
        "num_clips": len(clips),
        "sample_limit": limit,
        "trained_incidents": trained_incidents,
        "trained_cameras": trained_cameras,
        "warnings": warnings,
        "arms": entries,
        "scored_utc": runlog.utcnow(),
    }
    for arm, rows in arm_rows.items():
        arm_dir = out_dir / arm
        if arm_dir.exists():
            shutil.rmtree(arm_dir)
        write_rows(arm_dir / "frames.csv", rows, FRAME_FIELDS)
        inc = incident_rows(rows, params.modes)
        write_rows_plain(arm_dir / "incidents.csv", inc)
        (arm_dir / "summary.json").write_text(
            json.dumps(summarize(rows, params.modes), indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    report = render_report(manifest, arm_rows)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    if trained_incidents or trained_cameras:
        log.info("LEAKAGE: %d eval incidents / %d cameras are in this run's train "
                 "split", len(trained_incidents), len(trained_cameras))
    log.info("wrote %s", out_dir)
    return 0


def setup_logging(out_dir: Path):
    """stdout plus `<out_dir>/eval.log` — not runlog's `train.log`, which would
    read as a training log once finalize_run copies the directory."""
    import logging
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("smokeftv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(out_dir / "eval.log", mode="w")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def write_rows_plain(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, metavar="RUN_DIR",
                        help="a run directory, ckpts/<run>")
    parser.add_argument("--ckpt", default=None,
                        help="score this checkpoint instead of best.pt, e.g. "
                             "ckpts/<run>/last.pt")
    parser.add_argument("--arm-name", default=None,
                        help="name for --ckpt's arm (default: from the filename)")
    parser.add_argument("--mode", action="append", default=None,
                        choices=["prompt_every_frame", "keyframe", "conditional"],
                        help="restrict to these inference modes (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N clips — smoke test, not a "
                             "result, and never cached")
    parser.add_argument("--refresh", action="store_true",
                        help="rescore both arms even if they are cached")
    parser.add_argument("--box-jitter", default=None, metavar="PMIN,PMAX",
                        help="pad every prompt box by p ~ U(PMIN, PMAX) per axis "
                             "(seeded per frame, identical for every run) and "
                             "write gt_test_set_boxjitter/ instead")
    args = parser.parse_args()
    jitter = None
    if args.box_jitter:
        low, high = (float(v) for v in args.box_jitter.split(","))
        jitter = (low, high)
    return build(Path(args.run), ckpt_request=args.ckpt, arm_name=args.arm_name,
                 modes=tuple(args.mode) if args.mode else None, limit=args.limit,
                 refresh=args.refresh, box_jitter=jitter)


if __name__ == "__main__":
    raise SystemExit(main())
