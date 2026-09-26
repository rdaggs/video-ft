#!/usr/bin/env bash
# Regenerate this run's corpus, or rerun it. Written by scripts/train.py.
#
# `repro.sh manifest` rebuilds the clip list from this run's OWN
# config_resolved.yaml and diffs it against what the run actually built. A clean
# diff means the corpus is still a function of (config, data, seed) and nothing
# else. The only thing that can make it dirty is annotations.json or
# polygons.coco.json changing on disk — which is exactly what it is for.
set -euo pipefail

REPO=/home/rileydaggs/projects/video-ft
RUN=20260925_183517Z_run0
TAG=run0
GIT_SHA=0151687
cd "$REPO"

HERE="/home/rileydaggs/projects/video-ft/ckpts/20260925_183517Z_run0"
NOW="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
[[ "$NOW" == "$GIT_SHA" ]] || \
  echo "repro.sh: repo is at $NOW, this run was $GIT_SHA" >&2

case "${1:-manifest}" in
  manifest)
    .venv/bin/python -m scripts.build_clips \
      --config "$HERE/config_resolved.yaml" \
      --split train --out "$HERE/repro_manifest.jsonl"
    .venv/bin/python -m scripts.build_clips --diff \
      "$HERE/clip_manifest.jsonl" "$HERE/repro_manifest.jsonl"
    ;;
  train)
    ./train.sh --gpu "${2:?usage: repro.sh train <gpu>}" --tag "$TAG" \
      --config "$HERE/config_resolved.yaml" \
      > "logs/$(date -u +%Y%m%d_%H%M%SZ)_${TAG}-repro.log" 2>&1
    ;;
  *) echo "usage: repro.sh [manifest|train <gpu>]" >&2; exit 2 ;;
esac
