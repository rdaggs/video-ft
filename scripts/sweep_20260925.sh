#!/usr/bin/env bash
#
# sweep_20260925.sh — the 2026-09-25 Phase A batch, one command.
#
#   scripts/sweep_20260925.sh                 queue all of it on GPU 6, detached
#   scripts/sweep_20260925.sh --gpus 4,5,6    same queue, three GPUs pulling from it
#   scripts/sweep_20260925.sh --only boxjitter,control
#   scripts/sweep_20260925.sh --dry           print the plan and exit
#
# Every entry is `train.sh --finalize`, so each one ends as a pushed
# experiments/<run>/ with best.pt, both GT reports and the manifests. On one
# GPU they run back to back: ~4.5 h of training per 8-epoch run (~2.3 h for
# ep4) plus ~20 min of GT eval, so the whole queue is ~29 h. Runs start in the
# order listed, so the comparisons that matter most land first.
#
# The code is pinned to the commit the sweep starts at. Finalize commits only
# experiments/, so the tree stays clean; if anything OUTSIDE experiments/
# changes before a queued run starts, that worker halts rather than launch a
# run from code the launch commit does not describe. Restart the rest with
# --only.
#
# Stop everything: kill -- -<pgid>, printed at launch and in the sweep log.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# tag | overrides | hypothesis. Baseline for all of them is `control`, not run0:
# control carries the fused-metric selection and the conditional, GT-free val
# pass that run0 predates.
EXPERIMENTS=(
  "control||run0's recipe under honest selection (val iou_fused, conditional, no GT) — the bar"
  "boxjitter|prompt.box_jitter.enabled=true prompt.box_jitter.pad_min=0.5 prompt.box_jitter.pad_max=2.0|prompt boxes padded 50-200% per axis: robust to loose detector boxes; wins on gt_test_set_boxjitter, costs little on clean boxes"
  "lr3e-5|train.lr=3e-5|run0 peaked at epoch 3 at 1e-4 and then overfit; a lower lr peaks later and higher"
  "hardkeep|train.dropout.p_min=0.1 train.dropout.p_max=0.7|longer unprompted gaps make memory carry the object; closes keyframe/conditional toward prompt_every_frame"
  "reg|model.memory_attn_dropout=0.1 train.weight_decay=0.1|regularising the 7.52M memory path delays the post-epoch-3 overfit"
  "predmem|model.mem_mask_source=predicted_best|memory trained on the candidate inference writes (IoU head's pick), not the GT-picked one; removes a train/test mismatch"
  "ep4|train.epochs=4|run0's cosine schedule was still at 98% lr at its best epoch; a 4-epoch cosine anneals where the peak was"
)

GPUS="6"
ONLY=""
DRY=0
FOREGROUND=0
PIN=""
SWEEP_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)         GPUS="$2"; shift 2 ;;
    --only)         ONLY="$2"; shift 2 ;;
    --dry)          DRY=1; shift ;;
    --foreground)   FOREGROUND=1; shift ;;
    --pin)          PIN="$2"; shift 2 ;;
    --sweep-id)     SWEEP_ID="$2"; shift 2 ;;
    -h|--help)      sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *)              echo "sweep: unknown argument $1" >&2; exit 2 ;;
  esac
done

selected=()
for entry in "${EXPERIMENTS[@]}"; do
  tag="${entry%%|*}"
  if [[ -z "$ONLY" || ",$ONLY," == *",$tag,"* ]]; then
    selected+=("$entry")
  fi
done
[[ ${#selected[@]} -gt 0 ]] || { echo "sweep: --only $ONLY matches nothing" >&2; exit 2; }

if [[ $DRY -eq 1 ]]; then
  echo "GPUs ${GPUS}; ${#selected[@]} runs, in order:"
  for entry in "${selected[@]}"; do
    IFS='|' read -r tag overrides why <<<"$entry"
    printf '  %-10s %s\n             %s\n' "$tag" "${overrides:-<no overrides>}" "$why"
  done
  exit 0
fi

# --- launcher: check, pin, detach ------------------------------------------ #
if [[ $FOREGROUND -eq 0 ]]; then
  if ! git diff-index --quiet HEAD --; then
    echo "sweep: working tree is dirty; commit first so the sweep has a launch commit." >&2
    git status --short >&2
    exit 4
  fi
  git fetch -q origin && [[ -z "$(git log --oneline HEAD..origin/main 2>/dev/null)" ]] || {
    echo "sweep: origin/main has commits this checkout lacks; git pull first." >&2; exit 4; }
  [[ -z "$(git log --oneline origin/main..HEAD 2>/dev/null)" ]] || {
    echo "sweep: HEAD is not pushed; push the launch commit first so the records link to it." >&2; exit 4; }
  for gpu in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
    if (( used > 1000 )); then
      echo "sweep: GPU $gpu already has ${used} MiB in use." >&2; exit 5
    fi
  done
  PIN="$(git rev-parse HEAD)"
  SWEEP_ID="$(date -u +%Y%m%d_%H%M%SZ)_sweep0925"
  mkdir -p logs
  setsid nohup "$0" --foreground --gpus "$GPUS" --pin "$PIN" --sweep-id "$SWEEP_ID" \
    ${ONLY:+--only "$ONLY"} > "logs/${SWEEP_ID}.log" 2>&1 < /dev/null &
  pid=$!
  sleep 1
  echo "sweep: ${#selected[@]} runs on GPU(s) ${GPUS}, code pinned at ${PIN:0:7}"
  echo "  log   logs/${SWEEP_ID}.log   (each run also gets logs/<run>.log)"
  echo "  stop  kill -- -${pid}"
  exit 0
fi

# --- the queue ------------------------------------------------------------- #
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
QUEUE="$(mktemp "/tmp/${SWEEP_ID}.queue.XXXX")"
printf '%s\n' "${selected[@]}" > "$QUEUE"
GIT_LOCK="$REPO_ROOT/.git/finalize.lock"
say "sweep ${SWEEP_ID} pgid $$ pin ${PIN:0:7} gpus ${GPUS}; stop with: kill -- -$$"

pop() {
  # One entry off the top, under a lock, so GPUs share the queue.
  (
    flock 8
    head -n 1 "$QUEUE"
    sed -i '1d' "$QUEUE"
  ) 8>"${QUEUE}.lock"
}

worker() {
  local gpu="$1" entry tag overrides why log status
  while entry="$(pop)" && [[ -n "$entry" ]]; do
    IFS='|' read -r tag overrides why <<<"$entry"
    if ! git diff --quiet "$PIN" HEAD -- . ':(exclude)experiments' \
        || ! git diff-index --quiet HEAD --; then
      say "gpu $gpu HALT before $tag: code moved since ${PIN:0:7} or the tree is dirty."
      say "  restart the rest from a clean commit: scripts/sweep_20260925.sh --gpus $gpu --only <tags>"
      return 1
    fi
    log="logs/$(date -u +%Y%m%d_%H%M%SZ)_${tag}.log"
    say "gpu $gpu START $tag -> $log  [${overrides:-no overrides}]"
    # train.sh refuses a dirty tree; hold finalize's lock across its gate so a
    # neighbour's commit cannot stage files under it.
    exec 9>"$GIT_LOCK"
    flock 9
    # shellcheck disable=SC2086
    ./train.sh --gpu "$gpu" --tag "$tag" --finalize $overrides > "$log" 2>&1 &
    local child=$!
    for _ in $(seq 120); do
      grep -q '^run_name' "$log" 2>/dev/null && break
      kill -0 "$child" 2>/dev/null || break
      sleep 1
    done
    flock -u 9
    exec 9>&-
    set +e
    wait "$child"; status=$?
    set -e
    say "gpu $gpu END   $tag exit $status  ($log)"
    sleep 5
  done
}

trap 'say "sweep: signal; stopping"; kill -- -$$ 2>/dev/null' TERM INT HUP
pids=()
for gpu in ${GPUS//,/ }; do
  worker "$gpu" &
  pids+=($!)
  sleep 2
done
set +e
for pid in "${pids[@]}"; do wait "$pid"; done
say "sweep ${SWEEP_ID} done; remaining queue: $(wc -l < "$QUEUE") entries"
rm -f "$QUEUE" "${QUEUE}.lock"
