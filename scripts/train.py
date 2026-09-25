"""Train the SAM 3 video tracker's memory path on smoke masklets.

Invoked by `train.sh`, which owns the run name: it resolves it from its own
stdout so `ckpts/<run>/` and `logs/<run>.log` cannot drift apart. Everything a
run varies lives in the config, not here.

`--dry-run` stops after the run directory is complete and the corpus is built.
That is the whole of build step 1: a well-formed run directory, a reproducible
manifest, and `repro.sh`, with no GPU touched.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smokeftv import config_all, runlog                         # noqa: E402
from smokeftv.clips import clip_signature, manifest_row         # noqa: E402
from smokeftv.config import load_config, save_config            # noqa: E402
from smokeftv.corpus import build_clips                         # noqa: E402
from smokeftv.prompting import frame_keep_mask, schedule_row    # noqa: E402

# What engine.run_epoch returns. iou_best is the GT-picked candidate: an oracle,
# logged so the gap to iou_fused stays visible, never a sensible selector.
VAL_METRICS = ("iou_fused", "iou_selected", "iou_best")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--ckpt-dir", required=True, type=Path)
    parser.add_argument("--tag", default="")
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--git-dirty", default="0")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the run directory and the corpus, then stop")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main() -> int:
    args = parse_args()
    # `run_name` is applied AFTER the passthrough overrides so the launcher
    # wins: train.sh has already symlinked ckpts/<run>/run.log at
    # logs/<run>.log, and a run that renamed itself would leave it dangling.
    cfg = load_config(args.config, [*args.overrides, f"run_name={args.run_name}"])

    run_dir = cfg.run_dir
    if run_dir.resolve() != args.ckpt_dir.resolve():
        raise SystemExit(
            f"--ckpt-dir {args.ckpt_dir} but run.ckpts_root/{cfg.run_name} is "
            f"{run_dir}. train.sh has already symlinked {args.ckpt_dir}/run.log "
            "at logs/<run>.log; writing elsewhere would leave a run directory "
            "whose log points at another run. Fix run.ckpts_root, or the "
            "launcher.")
    # Here and not in config.py: eval_video loads older runs' resolved configs,
    # which say iou_polygon, and those must stay loadable.
    if cfg.train.select_metric not in VAL_METRICS:
        raise SystemExit(
            f"train.select_metric is {cfg.train.select_metric!r} but the val "
            f"pass computes only {list(VAL_METRICS)}; best.pt would be selected "
            "on a number that is never measured.")

    log = runlog.setup_logging(run_dir)
    seed_everything(cfg.seed)

    # FIRST, before a single JPEG is opened: a run that dies in the corpus scan
    # still records what it was trying to do.
    save_config(cfg, run_dir / "config_resolved.yaml")
    splits = runlog.write_splits_resolved(run_dir / "splits_resolved.yaml", cfg)
    if splits["leaked_cameras"]:
        raise SystemExit(f"train and held-out splits share cameras "
                         f"{splits['leaked_cameras']}")

    log.info("run %s  tag=%s  git=%s%s", cfg.run_name, args.tag, args.git_sha,
             " (DIRTY)" if args.git_dirty == "1" else "")
    log.info("config %s -> %s", args.config, run_dir / "config_resolved.yaml")
    log.info("splits: train %d incidents / %d cameras, val %d, test %d",
             len(splits["train_incidents"]), len(splits["train_cameras"]),
             len(splits["val_incidents"]), len(splits["test_incidents"]))

    problems = [m for sev, m in config_all.check() if sev == "blocking"]
    if problems:
        for message in problems:
            log.info("config_all --check: %s", message)
        raise SystemExit("config_all --check failed; see above")
    log.info("config_all --check: clean (%d divergences still true)",
             len(config_all.load().get("divergences", [])))

    log.info("building clips ...")
    train_clips, train_reports = build_clips(cfg, cfg.data.train_incidents, "train")
    val_clips, _ = build_clips(cfg, cfg.data.val_incidents, "val")
    for report in train_reports:
        if not report.clips:
            log.info("  %s", report)
    log.info("train: %d clips over %d incidents | val: %d clips",
             len(train_clips), sum(1 for r in train_reports if r.clips), len(val_clips))
    if not train_clips:
        raise SystemExit("no training clips; check clips.* and the data root")
    signature = clip_signature(train_clips)
    log.info("clip signature %s", signature[:16])

    if cfg.run.save_clip_manifest:
        manifest = runlog.JsonlLog(run_dir / "clip_manifest.jsonl", append=False)
        manifest.write_all(manifest_row(c, "train") for c in train_clips)
        manifest.write_all(manifest_row(c, "val") for c in val_clips)
        manifest.close()

    if cfg.run.save_prompt_schedule:
        schedule = runlog.JsonlLog(run_dir / "prompt_schedule.jsonl", append=False)
        for epoch in range(cfg.train.epochs):
            for clip in train_clips:
                mask, p = frame_keep_mask(len(clip.frame_indices), cfg.train.dropout,
                                          epoch, cfg.data.clips.seed, clip.clip_id)
                schedule.write(schedule_row(epoch, clip.clip_id, p, mask))
        schedule.close()

    runlog.write_repro(run_dir / "repro.sh", repo_root=REPO_ROOT,
                       run_name=cfg.run_name, tag=args.tag or cfg.run_name,
                       git_sha=args.git_sha, ckpt_dir=run_dir)

    snapshot = REPO_ROOT / "training_config.yaml"
    provenance_path = run_dir / "provenance.json"
    runlog.write_provenance(
        provenance_path,
        run_name=cfg.run_name, tag=args.tag, status="running",
        started_utc=runlog.utcnow(), finished_utc=None,
        argv=sys.argv, config_path=str(args.config), overrides=args.overrides,
        ckpt_dir=str(run_dir), log_path=f"{cfg.run.logs_root}/{cfg.run_name}.log",
        git={**runlog.git_info(REPO_ROOT), "sha": args.git_sha,
             "dirty": args.git_dirty == "1", "source": "train.sh"},
        sam3=runlog.git_info(REPO_ROOT / "sam3"),
        snapshot={"path": snapshot.name, "sha256": runlog.sha256_file(snapshot),
                  "divergences": [d["id"] for d in config_all.load()["divergences"]]},
        init_from={"path": config_all.load()["meta"]["init_from"],
                   "sha256": runlog.sha256_file(config_all.load()["meta"]["init_from"])},
        code=runlog.code_fingerprints(REPO_ROOT / "smokeftv"),
        geometry={"vendored": runlog.sha256_file(
            REPO_ROOT / "vendor" / "smokeseg" / "geometry.py", short=True)},
        env=runlog.env_fingerprint(),
        corpus={"data_root": cfg.data.root, "clip_seed": cfg.data.clips.seed,
                "clip_signature": signature,
                "train_clips": len(train_clips), "val_clips": len(val_clips),
                "train_incidents": len(cfg.data.train_incidents)},
    )
    log.info("run directory complete: %s", run_dir)
    for name in sorted(p.name for p in run_dir.iterdir()):
        log.info("  %s", name)

    if args.dry_run:
        runlog.update_provenance(provenance_path, status="dry-run",
                                 finished_utc=runlog.utcnow())
        log.info("--dry-run: stopping before the model is built")
        return 0

    # ---------------------------------------------------------------- model #
    import torch
    from torch.utils.data import DataLoader

    from smokeftv import checkpoint as ckpt
    from smokeftv.dataset import ClipDataset, collate
    from smokeftv.engine import build_scheduler, run_epoch
    from smokeftv.losses import MaskLoss
    from smokeftv.model import (build_tracker, param_groups,
                                prepare_tracker_for_training, set_trainable)

    log.info("building tracker ...")
    tracker = build_tracker(cfg.device)

    # Weights BEFORE the datasets: a bad overlay costs a second, a CLAHE pass
    # over the corpus costs minutes.
    init_from = [cfg.train.init_ckpt] if isinstance(cfg.train.init_ckpt, str) \
        else list(cfg.train.init_ckpt or [])
    init_from = init_from or [config_all.load()["meta"]["init_from"]]
    # Applied on resume too. A checkpoint holds only requires_grad tensors, so
    # last.pt carries the memory path and NOT L4E-2's trunk; skipping init_from
    # here would resume onto released SAM 3 weights with no error. last.pt is
    # loaded on top below, so it still wins wherever the two overlap.
    for path, meta in (ckpt.load_overlays(init_from, tracker) if init_from else []):
        log.info("init_from %s  %s", Path(path).name,
                 {k: meta[k] for k in ("run_name", "epoch") if k in meta})

    prepare_tracker_for_training(tracker, cfg, seed=cfg.seed)
    report = set_trainable(tracker, cfg)
    for name, row in report.items():
        if name != "_total":
            log.info("  trainable %-12s %4d tensors  %8.4fM",
                     name, row["tensors"], row["params"] / 1e6)
    log.info("  trainable %-12s %4d tensors  %8.4fM", "TOTAL",
             report["_total"]["tensors"], report["_total"]["params"] / 1e6)

    groups = param_groups(tracker, cfg)
    optimizer = torch.optim.AdamW(groups, lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    held = sum(len(g["params"]) for g in groups)
    log.info("optimizer: %s", ", ".join(
        f"{g['name']}={len(g['params'])}t@{g['lr']:.2e}" for g in groups))

    train_loader = DataLoader(ClipDataset(train_clips, cfg, "train"),
                              batch_size=1, shuffle=cfg.train.clip_policy == "shuffle",
                              num_workers=cfg.train.num_workers, collate_fn=collate,
                              persistent_workers=cfg.train.num_workers > 0)
    val_loader = DataLoader(ClipDataset(val_clips, cfg, "val"), batch_size=1,
                            shuffle=False, num_workers=cfg.train.num_workers,
                            collate_fn=collate,
                            persistent_workers=cfg.train.num_workers > 0)
    steps_per_epoch = max(len(train_loader) // cfg.train.grad_accum, 1)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    fingerprint = ckpt.resume_fingerprint(
        groups, steps_per_epoch=steps_per_epoch, epochs=cfg.train.epochs,
        lr_schedule=cfg.train.lr_schedule, train_clips=len(train_clips),
        select_metric=cfg.train.select_metric, clip_signature=signature,
        feature_cache_key="" if not cfg.feature_cache.enabled else signature)

    start_epoch, best_score, best_epoch = 1, -1.0, 0
    if cfg.train.resume:
        state = ckpt.read_resume(cfg.train.resume)
        # BEFORE a single tensor loads: a size mismatch would otherwise print
        # one line per adapter with the cause in none of them.
        ckpt.check_resume(state["fingerprint"], fingerprint)
        ckpt.load_overlays([Path(cfg.train.resume).with_name("last.pt")], tracker)
        optimizer.load_state_dict(state["optimizer"])
        if (scheduler is None) != (state["scheduler"] is None):
            raise SystemExit("scheduler presence changed across the resume")
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        ckpt.set_rng_state(state["rng"])
        best_score, best_epoch = state["best"]["score"], state["best"]["epoch"]
        start_epoch = state["epoch"] + 1
        log.info("resumed at epoch %d (best %.4f @ %d)", start_epoch,
                 best_score, best_epoch)

    runlog.update_provenance(
        provenance_path,
        model={"trainable_tensors": report["_total"]["tensors"],
               "trainable_params": report["_total"]["params"],
               "param_groups": [{"name": g["name"], "tensors": len(g["params"]),
                                 "lr": g["lr"]} for g in groups],
               "num_maskmem": tracker.num_maskmem,
               "max_cond_frames_in_attn": tracker.max_cond_frames_in_attn},
        resume_fingerprint=fingerprint)

    metrics = runlog.JsonlLog(run_dir / "metrics.jsonl")
    csv = runlog.CsvLog(run_dir / "log.csv")
    criterion = MaskLoss(cfg.train.loss)
    select = cfg.train.select_metric

    def record(split, epoch, stats, lr):
        row = {"epoch": epoch, "split": split, "t_utc": runlog.utcnow(), "lr": lr,
               "loss": round(stats["loss"], 5),
               **{m: round(stats[m], 5) for m in VAL_METRICS},
               "clips": stats["clips"], "secs": stats["secs"],
               "zero_grad_frames": stats["zero_grad_frames"]}
        metrics.write({**row, "iou_by_t": [round(v, 4) for v in stats["iou_by_t"]],
                       "iou_by_kind": {k: round(v, 4) if v == v else None
                                       for k, v in stats["iou_by_kind"].items()}})
        csv.write(row)
        return row

    # Epoch 0: the zero-shot bar, so every later number has something to beat
    # and a broken pipeline shows as an absurd epoch-0 IoU rather than as a
    # plausible epoch-1 one.
    if not cfg.train.resume:
        log.info("epoch 0: zero-shot val pass")
        stats = run_epoch(tracker, val_loader, cfg, epoch=0, train=False,
                          criterion=criterion, log_every=cfg.train.log_every,
                          probe_first=cfg.model.memory.log_bank_composition)
        record("val", 0, stats, cfg.train.lr)
        log.info("  val e0  loss %.4f iou_fused %.4f (best-of-3 %.4f)  by_t %s  "
                 "by_kind %s", stats["loss"], stats["iou"], stats["iou_best"],
                 [round(v, 3) for v in stats["iou_by_t"]],
                 {k: round(v, 3) for k, v in stats["iou_by_kind"].items() if v == v})
        best_score = stats[select]
        best_epoch = 0

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        tracker.train()
        if not cfg.model.train_mask_decoder:
            tracker.sam_mask_decoder.eval()
        if not cfg.model.train_prompt_encoder:
            tracker.sam_prompt_encoder.eval()
        tracker.backbone.eval()

        stats = run_epoch(tracker, train_loader, cfg, epoch=epoch,
                          optimizer=optimizer, scheduler=scheduler,
                          criterion=criterion, log_every=cfg.train.log_every,
                          train=True, probe_first=epoch == start_epoch and
                          cfg.model.memory.log_bank_composition)
        record("train", epoch, stats, lr)
        log.info("train e%d  loss %.4f iou_fused %.4f  %.0fs  zero-grad frames %d",
                 epoch, stats["loss"], stats["iou"], stats["secs"],
                 stats["zero_grad_frames"])

        if epoch % cfg.train.eval_every == 0 or epoch == cfg.train.epochs:
            tracker.eval()
            with torch.no_grad():
                vstats = run_epoch(tracker, val_loader, cfg, epoch=epoch,
                                   train=False, criterion=criterion,
                                   log_every=cfg.train.log_every)
            record("val", epoch, vstats, lr)
            log.info("  val e%d  loss %.4f iou_fused %.4f (best-of-3 %.4f)  "
                     "by_t %s  by_kind %s",
                     epoch, vstats["loss"], vstats["iou"], vstats["iou_best"],
                     [round(v, 3) for v in vstats["iou_by_t"]],
                     {k: round(v, 3) for k, v in vstats["iou_by_kind"].items()
                      if v == v})
            if vstats[select] > best_score:
                best_score, best_epoch = vstats[select], epoch
                ckpt.save(run_dir / "best.pt", tracker,
                          {"epoch": epoch, "run_name": cfg.run_name,
                           "select_metric": select, select: best_score})
                log.info("  new best %s %.4f -> best.pt", select, best_score)

        # last.pt and last_resume.pt back to back, so the two halves always
        # describe the same epoch. A resume that restored epoch N's optimizer
        # onto epoch N-1's weights is a silent half-step the fingerprint cannot
        # catch.
        ckpt.save(run_dir / "last.pt", tracker,
                  {"epoch": epoch, "run_name": cfg.run_name})
        if cfg.train.save_resume:
            ckpt.save_resume(run_dir / "last_resume.pt", optimizer=optimizer,
                             scheduler=scheduler, epoch=epoch,
                             best_score=best_score, best_epoch=best_epoch,
                             fingerprint=fingerprint)

    metrics.close()
    log.info("done. best %s %.4f at epoch %d", select, best_score, best_epoch)
    runlog.update_provenance(provenance_path, status="finished",
                             finished_utc=runlog.utcnow(),
                             best={"score": best_score, "epoch": best_epoch})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        # Tracebacks go to stderr, which train.log never sees. Mirror the cause
        # into it, and distinguish an interrupt — where the location it happened
        # to land on is noise — from a failure, where it is the whole point.
        import logging
        logger = logging.getLogger("smokeftv")
        if sys.exc_info()[0] is KeyboardInterrupt:
            logger.info("INTERRUPTED")
        else:
            logger.info("FAILED: %s", traceback.format_exc())
        raise
