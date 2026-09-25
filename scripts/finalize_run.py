#!/usr/bin/env python
"""Turn a finished run into a committed, pushed experiment record.

    python -m scripts.finalize_run --run 20260925_183517Z_run0

`train.sh --finalize` calls this when training exits 0, on the same GPU. By
hand, it is also how an interrupted or eval-only experiment gets recorded.

It does four things, in this order:

1. **Scores the GT test set** with `scripts/eval_video.py` unless
   `ckpts/<run>/gt_test_set/` already holds a full (non `--limit`) result.
2. **Copies the run into `experiments/<run>/`**: `best.pt`, and every other
   file in `ckpts/<run>/` except checkpoints and anything over
   `MAX_COPY_BYTES`, plus the launcher's log `logs/<run>.log` under its own
   tagged name. `last.pt` and `last_resume.pt` stay on /data3.
3. **Writes the two manifests** that make the record exact:
     checkpoints.json  every *.pt in the run dir: real path on /data3, bytes,
                       sha256, whether it is in git, and the overlays it
                       composes with
     data.json         per-incident sha256 of annotations.json and
                       polygons.coco.json for every train/val incident and the
                       GT set, so a relabel since the run is detectable
   and `EXPERIMENT.md`, the page you read on GitHub.
4. **Commits only `experiments/<run>/`** and pushes. Only that path, so whatever
   else is dirty in the tree is left alone. A push failure warns and keeps the
   local commit.

The code the run used is the commit `train.sh` recorded (`provenance.json`
git.sha), which the launch commit made clean — so the record links to it rather
than copying source.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS = REPO_ROOT / "experiments"
# prompt_schedule.jsonl is ~5 MB at 16 epochs and is the largest text artifact a
# run writes; anything bigger is a binary that belongs on /data3.
MAX_COPY_BYTES = 20 * 1024 * 1024
SKIP_SUFFIXES = {".pt", ".pth", ".ckpt", ".tmp"}
# The one checkpoint the record carries. It lives here rather than under ckpts/
# because ckpts is a symlink onto /data3 and git will not add a path behind a
# symlink. GitHub rejects files over 100 MB, so a bigger best.pt (an unfrozen
# encoder) stays on /data3 and is only hashed.
GIT_CKPT = "best.pt"
MAX_GIT_CKPT_BYTES = 95 * 1024 * 1024


def sh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True,
                          check=check)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run_eval(run_dir: Path, refresh: bool) -> tuple[str, str]:
    """`(status, note)`. Never raises: a failed eval is recorded, not fatal."""
    manifest = read_json(run_dir / "gt_test_set" / "manifest.json")
    scored = (manifest.get("arms") or [{}])[-1]
    ckpt, recorded = scored.get("ckpt"), (scored.get("overlay_sha256") or [None])[-1]
    # By hash, not by existence: an eval taken mid-run scored an earlier
    # best.pt, and training overwrites that file in place.
    if (manifest and not manifest.get("sample_limit") and not refresh and ckpt
            and Path(ckpt).is_file() and sha256(Path(ckpt)) == recorded):
        return "ok", "already scored"
    if not any((run_dir / f).is_file() for f in ("best.pt", "last.pt")):
        return "skipped", "no checkpoint in the run directory"
    if not (run_dir / "config_resolved.yaml").is_file():
        return "skipped", "no config_resolved.yaml"
    command = [sys.executable, str(REPO_ROOT / "scripts" / "eval_video.py"),
               "--run", str(run_dir)]
    if refresh:
        command.append("--refresh")
    print(f"finalize: scoring the GT test set: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        return "failed", f"eval_video.py exited {result.returncode}"
    return "ok", "scored just now"


def copy_run(run_dir: Path, exp_dir: Path) -> list[dict]:
    """Mirror the run dir minus binaries. Returns what was left behind."""
    skipped = []
    for src in sorted(run_dir.rglob("*")):
        rel = src.relative_to(run_dir)
        if src.is_dir():
            continue
        if src.is_symlink() and not src.exists():
            continue
        real = src.resolve()
        size = real.stat().st_size
        if str(rel) == GIT_CKPT:
            if size <= MAX_GIT_CKPT_BYTES:
                shutil.copy2(real, exp_dir / rel)
            else:
                print(f"finalize: WARNING {GIT_CKPT} is {size / 2**20:.0f} MB, over "
                      "GitHub's limit; it stays on /data3 (see checkpoints.json)",
                      file=sys.stderr)
            continue
        if src.suffix in SKIP_SUFFIXES or size > MAX_COPY_BYTES:
            if src.suffix not in (".pt", ".tmp"):
                skipped.append({"file": str(rel), "path": str(real), "bytes": size,
                                "sha256": sha256(real)})
            continue
        # run.log is a symlink to logs/<run>.log, which is copied under its own
        # tagged name below; one copy is enough.
        if rel.name == "run.log" and src.is_symlink():
            continue
        dst = exp_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(real, dst)
    return skipped


def checkpoint_manifest(run_dir: Path, exp_dir: Path, provenance: dict) -> dict:
    ckpts = []
    for path in sorted(run_dir.glob("*.pt")):
        digest = sha256(path)
        copy = exp_dir / path.name
        entry = {"file": path.name, "path": str(path.resolve()),
                 "bytes": path.stat().st_size, "sha256": digest,
                 "in_git": (str(copy.relative_to(REPO_ROOT))
                            if copy.is_file() and sha256(copy) == digest else None)}
        try:
            import torch
            blob = torch.load(path, map_location="cpu", weights_only=False)
            meta = blob.get("meta") if isinstance(blob, dict) else None
            if meta:
                entry["meta"] = {k: v for k, v in meta.items()
                                 if isinstance(v, (str, int, float, bool))}
            elif isinstance(blob, dict) and "epoch" in blob:
                entry["meta"] = {"epoch": blob["epoch"], "kind": "resume state"}
        except Exception as exc:                          # noqa: BLE001
            entry["meta_error"] = str(exc)[:200]
        ckpts.append(entry)
    return {
        "note": "best.pt is committed beside this file (`in_git`); the rest stay "
                "on /data3 at `path`. A checkpoint is a sparse overlay of "
                "requires_grad tensors and loads ON TOP of init_from (see "
                "smokeftv.checkpoint).",
        "init_from": provenance.get("init_from"),
        "checkpoints": ckpts,
    }


def data_manifest(run_dir: Path) -> dict:
    import yaml
    splits = yaml.safe_load((run_dir / "splits_resolved.yaml").read_text()) \
        if (run_dir / "splits_resolved.yaml").is_file() else {}
    provenance = read_json(run_dir / "provenance.json")

    def hashes(root: str, incidents) -> dict:
        out = {}
        for name in incidents:
            entry = {}
            for fname in ("annotations.json", "polygons.coco.json"):
                path = Path(root) / name / fname
                entry[fname] = sha256(path)[:16] if path.is_file() else None
            out[name] = entry
        return out

    gt_root = splits.get("gt_dataset_root")
    gt_incidents = sorted(p.name for p in Path(gt_root).iterdir()
                          if (p / "polygons.coco.json").is_file()) \
        if gt_root and Path(gt_root).is_dir() else []
    return {
        "data_root": splits.get("data_root"),
        "corpus": provenance.get("corpus"),
        "train": hashes(splits.get("data_root", ""), splits.get("train_incidents", [])),
        "val": hashes(splits.get("data_root", ""), splits.get("val_incidents", [])),
        "gt_dataset_root": gt_root,
        "gt": hashes(gt_root, gt_incidents) if gt_root else {},
        "held_out_cameras": splits.get("held_out_cameras"),
        "leaked_cameras": splits.get("leaked_cameras"),
    }


def val_table(run_dir: Path) -> str:
    path = run_dir / "log.csv"
    if not path.is_file():
        return "(no log.csv)"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if rows and not legacy_log(run_dir):
        lines = ["| epoch | split | loss | iou_fused | iou_selected | best-of-3 "
                 "| zero_grad_frames | secs |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r.get('epoch')} | {r.get('split')} | {r.get('loss')} | "
                         f"{r.get('iou_fused')} | {r.get('iou_selected')} | "
                         f"{r.get('iou_best')} | {r.get('zero_grad_frames')} | "
                         f"{r.get('secs')} |")
        return "\n".join(lines)
    lines = ["The `iou_fused` column of this run's log.csv is really best-of-3 "
             "(the GT-picked candidate, an oracle); the fused mask was not "
             "logged. See gt_test_set/ for fused numbers.", "",
             "| epoch | split | loss | best-of-3 | zero_grad_frames | secs |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.get('epoch')} | {r.get('split')} | {r.get('loss')} | "
                     f"{r.get('iou_fused')} | {r.get('zero_grad_frames')} | "
                     f"{r.get('secs')} |")
    return "\n".join(lines)


def select_metric(run_dir: Path) -> str:
    import yaml
    try:
        cfg = yaml.safe_load((run_dir / "config_resolved.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        return "iou_fused"
    return (cfg.get("train") or {}).get("select_metric", "iou_fused")


def legacy_log(run_dir: Path) -> bool:
    """Runs before the clip_loss fix logged best-of-3 under the name iou_fused
    and have no iou_best column."""
    path = run_dir / "log.csv"
    if not path.is_file():
        return False
    header = next(csv.reader(path.open(encoding="utf-8")), [])
    return "iou_best" not in header


def launch_message(sha: str) -> str:
    if not sha or sha == "unknown":
        return ""
    result = sh("git", "log", "-1", "--format=%B", sha, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def remote_url() -> str:
    url = sh("git", "remote", "get-url", "origin", check=False).stdout.strip()
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:"):]
    return url.removesuffix(".git")


def experiment_md(run: str, run_dir: Path, eval_status: tuple[str, str],
                  skipped: list[dict], log_name: str | None) -> str:
    prov = read_json(run_dir / "provenance.json")
    git = prov.get("git") or {}
    sha = git.get("sha", "unknown")
    url = remote_url()
    code_link = f"[`{sha}`]({url}/tree/{sha})" if url and sha != "unknown" else f"`{sha}`"
    best = prov.get("best") or {}
    intent = launch_message(sha)
    argv = prov.get("argv") or []
    overrides = prov.get("overrides") or []
    report = run_dir / "gt_test_set" / "report.txt"
    lines = [
        f"# {prov.get('tag') or run}",
        "",
        f"- **run** `{run}`",
        f"- **status** {prov.get('status', 'unknown')}  "
        f"({prov.get('started_utc')} -> {prov.get('finished_utc')})",
        f"- **code** {code_link}{' (DIRTY at launch)' if git.get('dirty') else ''}"
        f"  sam3 `{(prov.get('sam3') or {}).get('sha')}`",
        f"- **config** `{prov.get('config_path')}` at that commit; overrides: "
        f"`{' '.join(overrides) or 'none'}`. The literal merged config is "
        "`config_resolved.yaml` here.",
        (f"- **best** val best-of-3 (oracle; logged as iou_fused) "
         f"{best.get('score')} at epoch {best.get('epoch')}"
         if legacy_log(run_dir) else
         f"- **best** val {select_metric(run_dir)} {best.get('score')} "
         f"at epoch {best.get('epoch')}"),
        ("- **checkpoints** `best.pt` is committed here and loads on top of "
         "init_from; `last.pt` / `last_resume.pt` stay on /data3 — see "
         "`checkpoints.json`" if (EXPERIMENTS / run / GIT_CKPT).is_file() else
         "- **checkpoints** all stay on /data3 — see `checkpoints.json` for "
         "paths and sha256"),
        f"- **GT test set** {eval_status[0]} ({eval_status[1]})",
        f"- **host** {(prov.get('env') or {}).get('host')} "
        f"GPU {(prov.get('env') or {}).get('gpu_name')}",
    ]
    if log_name:
        lines.append(f"- **log** `{log_name}`")
    if intent:
        lines += ["", "## Intent (launch commit message)", "", "```", intent, "```"]
    tag = prov.get("tag") or "repro"
    # The record is committed AFTER the launch commit, so it does not exist once
    # that commit is checked out: take the config out first.
    lines += ["", "## Reproduce", "", "```bash",
              f"git show origin/main:experiments/{run}/config_resolved.yaml "
              f"> /tmp/{run}.yaml",
              f"git checkout {sha}",
              f"./train.sh --gpu <N> --tag {tag}-repro --config /tmp/{run}.yaml "
              f"> logs/$(date -u +%Y%m%d_%H%M%SZ)_{tag}-repro.log 2>&1 &",
              "```"]
    if argv:
        lines += ["", f"Original argv: `{' '.join(argv)}`"]
    lines += ["", "## Validation (log.csv)", "", val_table(run_dir)]
    if report.is_file():
        lines += ["", "## GT test set (gt_test_set/report.txt)", "", "```",
                  report.read_text(encoding="utf-8").rstrip(), "```"]
    if skipped:
        lines += ["", "## Left on /data3 (too large for git)", ""]
        lines += [f"- `{s['file']}` {s['bytes']:,} B sha256 `{s['sha256'][:16]}` "
                  f"at `{s['path']}`" for s in skipped]
    return "\n".join(lines) + "\n"


def commit_and_push(exp_dir: Path, run: str, tag: str, push: bool) -> None:
    rel = str(exp_dir.relative_to(REPO_ROOT))
    lock_path = REPO_ROOT / ".git" / "finalize.lock"
    with lock_path.open("w") as lock:
        # Two runs finishing together would otherwise race on .git/index.lock.
        fcntl.flock(lock, fcntl.LOCK_EX)
        sh("git", "add", "--force", "--", rel)
        staged = sh("git", "diff", "--cached", "--quiet", "--", rel, check=False)
        if staged.returncode == 0:
            print(f"finalize: nothing new to commit under {rel}")
        else:
            message = f"results({tag}): {run}\n\nRecorded by scripts/finalize_run.py."
            sh("git", "commit", "--only", "-m", message, "--", rel)
            print(f"finalize: committed {rel} as "
                  f"{sh('git', 'rev-parse', '--short', 'HEAD').stdout.strip()}")
        if not push:
            return
        branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        if branch == "HEAD":
            print("finalize: WARNING detached HEAD; not pushing", file=sys.stderr)
            return
        result = sh("git", "push", "origin", branch, check=False)
        if result.returncode != 0:
            print(f"finalize: WARNING push failed; the commit is kept locally.\n"
                  f"{result.stderr.strip()}\n  retry: git pull --rebase && "
                  f"git push origin {branch}", file=sys.stderr)
        else:
            print(f"finalize: pushed {branch} to origin")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True,
                        help="run name (ckpts/<run>) or a path to the run dir")
    parser.add_argument("--no-eval", action="store_true",
                        help="record without scoring the GT test set")
    parser.add_argument("--refresh-eval", action="store_true",
                        help="rescore the GT test set even if already scored")
    parser.add_argument("--no-push", action="store_true",
                        help="commit locally only")
    parser.add_argument("--no-commit", action="store_true",
                        help="write experiments/<run>/ and stop")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = REPO_ROOT / "ckpts" / args.run
    run_dir = run_dir if run_dir.is_absolute() else (REPO_ROOT / run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"no run directory at {run_dir}")
    run = run_dir.name
    provenance = read_json(run_dir / "provenance.json")
    tag = provenance.get("tag") or run
    if provenance.get("status") == "running":
        print("finalize: WARNING provenance says status=running — the run is "
              "still going or died before its handler", file=sys.stderr)

    eval_status = ("skipped", "--no-eval") if args.no_eval else \
        run_eval(run_dir, args.refresh_eval)
    print(f"finalize: GT test set {eval_status[0]} ({eval_status[1]})", flush=True)

    exp_dir = EXPERIMENTS / run
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True)
    skipped = copy_run(run_dir, exp_dir)

    log_name = None
    log_path = REPO_ROOT / (provenance.get("log_path") or f"logs/{run}.log")
    if log_path.is_file():
        log_name = log_path.name
        shutil.copy2(log_path.resolve(), exp_dir / log_name)

    (exp_dir / "checkpoints.json").write_text(
        json.dumps(checkpoint_manifest(run_dir, exp_dir, provenance), indent=2) + "\n",
        encoding="utf-8")
    (exp_dir / "data.json").write_text(
        json.dumps(data_manifest(run_dir), indent=2) + "\n", encoding="utf-8")
    (exp_dir / "EXPERIMENT.md").write_text(
        experiment_md(run, run_dir, eval_status, skipped, log_name), encoding="utf-8")
    print(f"finalize: wrote {exp_dir.relative_to(REPO_ROOT)}")

    if not args.no_commit:
        commit_and_push(exp_dir, run, tag, push=not args.no_push)
    return 0 if eval_status[0] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
