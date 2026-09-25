"""Sparse-overlay checkpoints and the resume fingerprint.

A full SAM 3 state dict is ~3.4 GB; a Phase A checkpoint is 7.52M parameters.
So a checkpoint here stores **only `requires_grad` tensors**, addressed by full
parameter name, and overlays compose: L4E-2 owns `backbone.vision_backbone.*`
and a memory checkpoint owns `transformer.*` / `maskmem_backbone.*` / the token
Parameters, which are disjoint by construction.

Resume state is a **separate artifact**. AdamW's two moments per trainable
tensor would double the size of a file whose bytes get hashed for provenance
and whose contract is "a sparse overlay keyed by parameter name", so they go
beside it in `last_resume.pt`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch

RESUME_VERSION = 1


def trainable_state_dict(model) -> dict:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone()
            for k, v in model.state_dict().items() if k in names}


def save(path: str | Path, model, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trainable": trainable_state_dict(model), "meta": meta or {}}, path)
    return path


def read(path: str | Path) -> tuple[dict, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return blob["trainable"], blob.get("meta", {})


def _apply(model, state: dict) -> None:
    if not state:
        raise ValueError("checkpoint contains no tensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # load_state_dict copies in place, so parameter identity is unchanged and
    # autocast's identity-keyed cache would happily keep serving the weights
    # from before the load. Belt and braces; the cache should already be off.
    torch.clear_autocast_cache()
    if unexpected:
        raise ValueError(
            f"checkpoint has {len(unexpected)} key(s) this model does not: "
            f"{sorted(unexpected)[:4]}. A sparse overlay is addressed by full "
            "parameter name, so an unexpected key means it was trained against "
            "a different architecture.")


def load_overlays(paths: Sequence[str | Path], model) -> list[tuple[str, dict]]:
    """Apply several overlays, refusing on any key collision.

    Every file is read and collision-checked BEFORE any of them is applied, so
    a collision leaves the model at its released weights rather than half
    merged. Order is the order given, and it is not allowed to matter.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("load_overlays was given no checkpoints")
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError(f"the same checkpoint is listed twice: "
                         f"{[p.name for p in paths]}")
    loaded = [(p, *read(p)) for p in paths]
    owner: dict[str, Path] = {}
    for path, state, _ in loaded:
        clash = sorted(k for k in state if k in owner)
        if clash:
            other = owner[clash[0]]
            raise ValueError(
                f"{path.name} and {other.name} both contain {len(clash)} of the "
                f"same parameters (e.g. {clash[0]}). They are not composable "
                "overlays — train them on disjoint modules, or treat the later "
                "one as a SUCCESSOR that supersedes the earlier rather than "
                "listing both.")
        owner.update({k: path for k in state})
    for path, state, _ in loaded:
        _apply(model, state)
    return [(str(p), meta) for p, _, meta in loaded]


def rng_state(generator=None, clip_rng=None) -> dict:
    """Every stream that changes what the next epoch sees.

    The clip/prompt stream is this repo's version of the loader-generator trap:
    dropout is drawn per (epoch, clip), so restoring the global streams but not
    this one resumes epoch 9 onto epoch 1's dropout pattern — silent, and
    entirely plausible-looking.
    """
    import numpy as np
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "loader": generator.get_state() if generator is not None else None,
    }
    return state


def set_rng_state(state: dict, generator=None) -> None:
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        if state["cuda_devices"] != torch.cuda.device_count():
            raise ValueError(
                f"this run saw {state['cuda_devices']} CUDA device(s) and this "
                f"process sees {torch.cuda.device_count()} — "
                "CUDA_VISIBLE_DEVICES changed between the two halves of a run.")
        torch.cuda.set_rng_state_all(state["cuda"])
    if generator is not None and state.get("loader") is not None:
        generator.set_state(state["loader"])


def resume_fingerprint(groups, *, steps_per_epoch: int, epochs: int,
                       lr_schedule: str, train_clips: int, select_metric: str,
                       clip_signature: str, feature_cache_key: str = "") -> dict:
    """The facts a resume is not allowed to change.

    Deliberately not a checksum of the whole config — most of it can change
    harmlessly. These cannot:

    * **param_groups** — optimizer state is keyed by POSITION in its param list.
      A changed freeze policy reorders it, and if the counts happen to line up
      one tensor's momentum lands on another's.
    * **train_clips + clip_signature** — `train_incidents: auto` resolves at
      load time and the corpus grows while runs are in flight. The signature is
      there because the same COUNT can come from different frames.
    * **epochs under a schedule** — a cosine's shape is a function of total
      steps; restoring its position into a different total steps the LR back up
      at the seam.
    * **select_metric** — `best_score` is a number on ONE surface, and
      iou_fused and iou_polygon have disagreed on the sign of a change.
    * **feature_cache_key** — resuming onto a different key is resuming onto
      different input features under the same run name.
    """
    return {
        "param_groups": [{"name": g.get("name", str(i)),
                          "tensors": len(g["params"]),
                          "numel": sum(p.numel() for p in g["params"])}
                         for i, g in enumerate(groups)],
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "lr_schedule": lr_schedule,
        "train_clips": train_clips,
        "select_metric": select_metric,
        "clip_signature": clip_signature,
        "feature_cache_key": feature_cache_key,
    }


def check_resume(saved: dict, current: dict) -> None:
    if saved["param_groups"] != current["param_groups"]:
        raise ValueError(
            f"the freeze policy changed since this run started.\n"
            f"  saved:   {saved['param_groups']}\n"
            f"  current: {current['param_groups']}\n"
            "Optimizer state is keyed by position in its parameter list, so "
            "restoring it onto a different set puts one tensor's momentum on "
            "another's. Use that run's own config_resolved.yaml.")
    for key, why in (
        ("train_clips", "the corpus changed size — `train_incidents: auto` "
                        "resolves at load time and the labeling queue moves"),
        ("clip_signature", "the corpus is the same SIZE but built from "
                           "different frames"),
        ("steps_per_epoch", "the step budget per epoch changed"),
        ("select_metric", "best_score is a number on one surface; comparing "
                          "later epochs against a different one picks the "
                          "shipping checkpoint by comparing two metrics"),
        ("feature_cache_key", "the cached encoder features are not the ones "
                              "this run trained on"),
    ):
        if saved.get(key) != current.get(key):
            raise ValueError(
                f"{key}: saved {saved.get(key)!r}, current {current.get(key)!r} "
                f"— {why}. Pin the corpus from the original run's "
                "config_resolved.yaml, or start a fresh run.")
    if current["lr_schedule"] != "none" and saved["epochs"] != current["epochs"]:
        raise ValueError(
            f"train.epochs changed from {saved['epochs']} to {current['epochs']} "
            f"under lr_schedule {current['lr_schedule']!r}. A cosine's shape is "
            "a function of total step count; restoring its position into a "
            "different total steps the LR back up at the seam.")


def save_resume(path: str | Path, *, optimizer, scheduler, epoch: int,
                best_score: float, best_epoch: int, fingerprint: dict,
                generator=None) -> Path:
    path = Path(path)
    torch.save({
        "version": RESUME_VERSION,
        "epoch": epoch,
        "best": {"score": best_score, "epoch": best_epoch},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "fingerprint": fingerprint,
        "rng": rng_state(generator),
    }, path)
    return path


def read_resume(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Use train.init_ckpt to warm-start from "
            "those weights with a fresh optimizer instead.")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("version") != RESUME_VERSION:
        raise ValueError(
            f"{path} is resume version {blob.get('version')}, this is "
            f"{RESUME_VERSION}. Half-restoring an optimizer is worse than "
            "refusing to.")
    return blob
