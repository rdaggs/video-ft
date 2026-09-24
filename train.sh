#!/usr/bin/env bash
#
# train.sh — launcher for sam3-video-cvf.
#
#   ./train.sh --gpu 7 --tag L4E-2 > logs/$(date -u +%Y%m%d_%H%M%SZ)_L4E-2.log 2>&1 &
#
# The timestamp is computed by the INVOKING shell, so this script cannot
# recompute it without drifting by however long startup takes. It resolves its
# own stdout instead, so ckpts/<run_name>/ and logs/<run_name>.log always share
# a name. If stdout is not a .log file (interactive run) it falls back to
# generating its own timestamp.
#
# Extra args are passed through as config overrides:
#   ./train.sh --gpu 7 --tag lr3e-5 train.lr=3e-5 clips.stride=4 > ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

GPU=""
TAG=""
CONFIG="configs/train_video.yaml"
ALLOW_DIRTY=0
OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)         GPU="$2"; shift 2 ;;
    --tag)         TAG="$2"; shift 2 ;;
    --config)      CONFIG="$2"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    -h|--help)
      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0 ;;
    *)             OVERRIDES+=("$1"); shift ;;
  esac
done

[[ -n "$GPU" ]] || { echo "train.sh: --gpu is required" >&2; exit 2; }
[[ -n "$TAG" ]] || { echo "train.sh: --tag is required" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "train.sh: no such config: $CONFIG" >&2; exit 2; }

# --- run name: basename of our own stdout, minus .log ---------------------- #
resolve_run_name() {
  local target
  if target="$(readlink -f /proc/self/fd/1 2>/dev/null)" && [[ "$target" == *.log ]]; then
    basename "$target" .log
  else
    echo "$(date -u +%Y%m%d_%H%M%SZ)_${TAG}"
  fi
}

RUN_NAME="$(resolve_run_name)"
CKPT_DIR="ckpts/${RUN_NAME}"
LOG_PATH="logs/${RUN_NAME}.log"

if [[ -e "$CKPT_DIR" ]]; then
  echo "train.sh: ${CKPT_DIR} already exists — refusing to overwrite a run" >&2
  echo "          (resuming is scripts/train.py --resume ${CKPT_DIR})" >&2
  exit 3
fi

# --- reproducibility gate -------------------------------------------------- #
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY=0
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
  GIT_DIRTY=1
  if [[ "$ALLOW_DIRTY" -eq 0 ]]; then
    echo "train.sh: working tree is dirty. Commit, or pass --allow-dirty." >&2
    git status --short >&2
    exit 4
  fi
  echo "train.sh: WARNING running with a dirty tree (--allow-dirty)" >&2
fi

mkdir -p "$CKPT_DIR" logs

# ckpts/<run>/run.log -> logs/<run>.log, so a run directory is self-navigating
ln -sfn "../../${LOG_PATH}" "${CKPT_DIR}/run.log"

# --- banner: everything needed to identify this run, at the top of the log -- #
cat <<EOF
================================================================================
run_name   ${RUN_NAME}
tag        ${TAG}
gpu        ${GPU}
config     ${CONFIG}
overrides  ${OVERRIDES[*]:-<none>}
ckpt_dir   ${CKPT_DIR}
log        ${LOG_PATH}
git        ${GIT_SHA}$([[ $GIT_DIRTY -eq 1 ]] && echo ' (DIRTY)')
started    $(date -u +%Y-%m-%dT%H:%M:%SZ)
host       $(hostname)
================================================================================
EOF

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export SAM3_VIDEO_RUN_NAME="$RUN_NAME"

# scripts/train.py writes config_resolved.yaml (extends stripped),
# provenance.json, splits_resolved.yaml, clip_manifest.jsonl,
# prompt_schedule.jsonl, metrics.jsonl and repro.sh into CKPT_DIR.
exec python -m scripts.train \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --ckpt-dir "$CKPT_DIR" \
  --tag "$TAG" \
  --git-sha "$GIT_SHA" \
  --git-dirty "$GIT_DIRTY" \
  "${OVERRIDES[@]}"

