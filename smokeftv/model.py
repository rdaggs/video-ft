"""Building the SAM 3 tracker for training, and deciding what moves.

`build_tracker(with_backbone=True)` gives a standalone `Sam3TrackerPredictor`
that owns its own vision backbone — no detector needed, because we supply the
boxes ourselves. Its parameter namespace is `backbone.vision_backbone.*`, which
is *exactly* L4E-2's, so the image checkpoint overlays with no key remap at all
(verified: 66/66 tensors, zero unexpected, zero shape mismatch).

Nine things about the vendored source make `.train()` alone insufficient, and
`prepare_tracker_for_training` handles each one. They are documented at the call
site rather than here, because each is a specific line in a specific file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from smokeftv.config import MEMORY_MODULES, MEMORY_TOKENS

log = logging.getLogger("smokeftv")


_MLP_PATCHED = False


def patch_vit_mlp() -> None:
    """Make the ViT's MLP differentiable, and dtype-honest. Idempotent.

    `vitdet.Mlp.forward` routes through `perflib.fused.addmm_act`, which does
    two things that end an encoder finetune before it starts: it raises
    `ValueError("Expected grad to be disabled.")` when `torch.is_grad_enabled()`,
    and it `detach()`es `fc1.weight`, so even without the raise that projection
    could never receive a gradient. The raise tests the GLOBAL grad flag rather
    than the parameters, so it fires for every block once grad is on — the trunk
    cannot run with grad at all until this is replaced.

    It also casts to bf16 unconditionally while `fc2` respects autocast, so
    running the backbone outside autocast raises `mat1 and mat2 must have the
    same dtype`. That is why `engine._prepare_feats` runs under the same
    autocast as the rollout rather than in bare fp32.

    The grad-disabled branch is the original call, unchanged, so inference and
    frozen-encoder training take exactly the path they took before this existed.
    """
    global _MLP_PATCHED
    if _MLP_PATCHED:
        return
    from sam3.model import vitdet
    from sam3.perflib.fused import addmm_act

    def forward(self, x):
        if torch.is_grad_enabled():
            # addmm_act fuses (bias + x @ W.T) then the activation; fc1 followed
            # by act is the same composition, with a backward pass.
            x = self.act(self.fc1(x))
        else:
            x = addmm_act(type(self.act), self.fc1, x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)

    vitdet.Mlp.forward = forward
    _MLP_PATCHED = True


def load_base_weights(tracker, checkpoint_path: str | Path | None = None) -> dict:
    """Released SAM 3 weights into a standalone `build_tracker` instance.

    **`build_tracker` builds the architecture and nothing else.** Without this
    the tracker is randomly initialised, which does not crash and does not look
    obviously wrong — it produces near-full-frame masks with `iou_head` ~0.5 and
    `object_score_logits` ~0, i.e. exactly what an untrained head outputs. That
    reads as "the memory path does not work yet" rather than as "there are no
    weights", which is why this function is loud about its counts.

    `sam3.pt` is keyed for the full detector+tracker model:

        tracker.*            -> this tracker's own modules
        detector.backbone.*  -> the vision backbone we asked build_tracker for

    The rest of `detector.*` is the text encoder and the detection head, which a
    box-prompted tracker never calls.
    """
    if checkpoint_path is None:
        from sam3.model_builder import download_ckpt_from_hf
        checkpoint_path = download_ckpt_from_hf(version="sam3")
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in blob and isinstance(blob["model"], dict):
        blob = blob["model"]

    state = {k[len("tracker."):]: v for k, v in blob.items()
             if k.startswith("tracker.")}
    state.update({k[len("detector."):]: v for k, v in blob.items()
                  if k.startswith("detector.backbone.")})
    missing, unexpected = tracker.load_state_dict(state, strict=False)
    own = {n for n, _ in tracker.named_parameters()}
    unloaded = sorted(n for n in own if n not in state)
    if unloaded:
        raise RuntimeError(
            f"{len(unloaded)} tracker parameter(s) got no released weights, "
            f"e.g. {unloaded[:4]}. A partially initialised tracker produces "
            "plausible-looking garbage rather than an error.")
    log.info("base weights: %d tensors from %s (%d unexpected)",
             len(state), Path(checkpoint_path).name, len(unexpected))
    return {"path": str(checkpoint_path), "tensors": len(state)}


def build_tracker(device: str = "cuda", checkpoint_path: str | Path | None = None):
    """The tracker, with its backbone and its released weights, out of autocast."""
    # Before any instance exists, so none is ever built on the raising version.
    patch_vit_mlp()
    import sam3.model_builder as mb
    tracker = mb.build_tracker(apply_temporal_disambiguation=False,
                               with_backbone=True)
    _exit_global_autocast(tracker)
    load_base_weights(tracker, checkpoint_path)
    return tracker.to(device)


def _exit_global_autocast(tracker) -> None:
    """`Sam3TrackerPredictor.__init__` enters a process-global bf16 autocast and
    never exits it (`sam3_tracking_predictor.py:50-51`).

    Leaving it entered would put the loss reduction, the IoU thresholding and
    `optimizer.step()` under an autocast nobody scoped. Worse, it is created
    with `cache_enabled=True`, whose weight cache is keyed by tensor identity
    and is never invalidated by an in-place `optimizer.step()` — so the forward
    would keep serving pre-update weights. No crash, real numbers, wrong label.
    """
    context = getattr(tracker, "bf16_context", None)
    if context is not None:
        try:
            context.__exit__(None, None, None)
        except RuntimeError:
            pass
        tracker.bf16_context = None
    if torch.is_autocast_enabled():
        raise RuntimeError(
            "still inside an autocast after build_tracker. Training would run "
            "under an unscoped autocast with cache_enabled=True.")


def prepare_tracker_for_training(tracker, cfg, *, seed: int = 0):
    """Install what the training path reads and inference never sets up."""
    mem = cfg.model.memory

    # (1) Two `self.training`-gated branches read attributes __init__ never
    # sets, so `.train()` raises AttributeError before it does anything:
    #   sam3_tracker_base.py:333  self.teacher_force_obj_scores_for_mem
    #   sam3_tracker_base.py:684  self.prob_to_dropout_spatial_mem / self.rng
    # `and` does not save you — self.training is True, so Python evaluates the
    # second operand.
    #
    # teacher_force_obj_scores_for_mem is turned ON deliberately when
    # force_obj_appearing is set: it makes `_forward_sam_heads` take
    # `is_obj_appearing` from the GT mask instead of from the frozen presence
    # head. When that head emits <= 0 the mask becomes a constant NO_OBJ_SCORE,
    # the frame contributes exactly zero gradient, obj_ptr collapses to
    # no_obj_ptr and the memory for that frame is poisoned.
    tracker.teacher_force_obj_scores_for_mem = bool(cfg.model.force_obj_appearing)
    tracker.prob_to_dropout_spatial_mem = 0.0
    import numpy as np
    tracker.rng = np.random.default_rng(seed)

    # (2) Cosmetic under our design — we never call the session layer — but set
    # so `forward_tracking` cannot disagree with the rollout if anyone does.
    tracker.add_all_frames_to_correct_as_cond = bool(
        mem.prompted_frames_are_conditioning)

    # (3) Memory bank shape. num_maskmem must be set BEFORE any forward:
    # maskmem_tpos_enc is a Parameter of shape (num_maskmem, 1, 1, mem_dim)
    # sized at construction and indexed [num_maskmem - t - 1].
    slots = tracker.maskmem_tpos_enc.shape[0]
    if mem.num_maskmem > slots:
        raise ValueError(
            f"model.memory.max_recent_frames implies num_maskmem "
            f"{mem.num_maskmem}, but maskmem_tpos_enc has {slots} slots. "
            "Shrinking is fine (the extra slots go unused); growing would need "
            "a resized Parameter and untrained positions.")
    tracker.num_maskmem = mem.num_maskmem
    tracker.max_cond_frames_in_attn = mem.max_cond_frames_in_attn
    tracker.keep_first_cond_frame = mem.keep_first_cond_frame

    # (4) `.train()` switches on 0.1 dropout across the whole memory attention:
    # 16 nn.Dropout modules and 8 RoPEAttention.dropout_p. Train mode is still
    # required — see (6) — so the dropout is switched off by hand instead. The
    # single-clip overfit test cannot reach loss < 0.01 with it on.
    n_drop = n_attn = 0
    for module in tracker.transformer.modules():
        if isinstance(module, nn.Dropout):
            module.p = float(cfg.model.memory_attn_dropout)
            n_drop += 1
        if hasattr(module, "dropout_p"):
            module.dropout_p = float(cfg.model.memory_attn_dropout)
            n_attn += 1
    if not n_drop and not n_attn:
        raise RuntimeError(
            "found no dropout in tracker.transformer — the memory attention "
            "was rebuilt and model.memory_attn_dropout now silently does "
            "nothing.")

    # (5) Gradient checkpointing over the 4 memory-attention layers. A real
    # built-in hook, consumed at decoder.py:738 as
    # `act_ckpt_enable=self.training and self.use_act_checkpoint`.
    tracker.transformer.encoder.use_act_checkpoint = bool(
        cfg.model.grad_checkpoint_memory_attn)

    # (6) train() LAST. The one branch it actually buys is
    # sam3_tracker_base.py:821: in EVAL mode the mask is hard-binarised with
    # `(x > 0).float()` before the memory encoder — zero gradient — and in train
    # mode it is a differentiable sigmoid. Our regime prompts most frames, so
    # eval mode would starve maskmem_backbone of gradient almost entirely.
    tracker.train()

    # (7) ...but the frozen heads go back to eval. mask_decoder.py:150 gates
    # `_dynamic_multimask_via_stability` on `not self.training`, so a frozen
    # decoder left in train mode takes a DIFFERENT code path than the val pass
    # — the two forwards would disagree while being reported as one model.
    if not cfg.model.train_mask_decoder:
        tracker.sam_mask_decoder.eval()
    if not cfg.model.train_prompt_encoder:
        tracker.sam_prompt_encoder.eval()
    tracker.backbone.eval()

    # (8) SimpleMaskEncoder is what lets the memory encoder be called with
    # image=None, and therefore what lets a feature cache skip the RGB frames.
    from sam3.model.sam3_tracker_base import SimpleMaskEncoder
    if not isinstance(tracker.maskmem_backbone, SimpleMaskEncoder):
        raise RuntimeError(
            "maskmem_backbone is not a SimpleMaskEncoder, so _encode_new_memory "
            "reads the RGB image and track_step_train's image=None is wrong.")

    log.info("tracker prepared: num_maskmem=%d max_cond=%d keep_first=%s "
             "dropout=%.2f (%d modules, %d attn) grad_ckpt=%s",
             tracker.num_maskmem, tracker.max_cond_frames_in_attn,
             tracker.keep_first_cond_frame, cfg.model.memory_attn_dropout,
             n_drop, n_attn, cfg.model.grad_checkpoint_memory_attn)
    return tracker


def trainable_prefixes(cfg) -> tuple[list[str], list[str]]:
    """(prefix rules, exact-name rules) for the current freeze policy."""
    prefixes: list[str] = []
    exact: list[str] = []
    for name in cfg.model.memory.train:
        prefixes.extend(MEMORY_MODULES[name])
        if name == "attention":
            # The five bare Parameters belong with the attention they feed.
            # Exact match, not startswith, or a future `no_mem_embed_v2` would
            # be swept in silently.
            exact.extend(MEMORY_TOKENS)
    if cfg.model.train_mask_decoder:
        prefixes.append("sam_mask_decoder.")
    if cfg.model.train_prompt_encoder:
        prefixes.append("sam_prompt_encoder.")
    return prefixes, exact


def set_trainable(tracker, cfg) -> dict:
    """Freeze everything, then re-enable by name. Returns a small report."""
    tracker.requires_grad_(False)
    prefixes, exact = trainable_prefixes(cfg)
    groups: dict[str, list[str]] = {}
    for name, param in tracker.named_parameters():
        hit = next((p for p in prefixes if name.startswith(p)), None)
        if hit is None and name in exact:
            hit = "tokens"
        if hit is not None:
            param.requires_grad_(True)
            groups.setdefault(hit, []).append(name)
    report = {k: {"tensors": len(v),
                  "params": sum(dict(tracker.named_parameters())[n].numel() for n in v)}
              for k, v in sorted(groups.items())}
    total = sum(g["params"] for g in report.values())
    if total == 0:
        raise ValueError(
            "the freeze policy leaves nothing trainable. model.memory.train is "
            f"{cfg.model.memory.train!r} and train_mask_decoder is "
            f"{cfg.model.train_mask_decoder}.")
    report["_total"] = {"tensors": sum(g["tensors"] for g in report.values()),
                        "params": total}
    return report


def param_groups(tracker, cfg) -> list[dict]:
    """AdamW groups. Group 0 is `memory`, so `log.csv`'s lr column keeps
    meaning `train.lr` no matter which later phase adds a group.

    `mask_downsample` is reachable only through `_use_mask_as_output`, which the
    rollout never calls, so its grad is always None. It is kept in the
    checkpoint for completeness but left OUT of the optimizer: a tensor that can
    never receive a gradient overstates the trainable-params line.
    """
    mem_prefixes = tuple(p for name in cfg.model.memory.train
                         for p in MEMORY_MODULES[name] if name != "mask_down")
    memory, decoder, encoder, inert = [], [], [], []
    for name, param in tracker.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("mask_downsample."):
            inert.append(name)
        elif name.startswith(mem_prefixes) or name in MEMORY_TOKENS:
            memory.append(param)
        elif name.startswith("sam_mask_decoder.") or name.startswith("sam_prompt_encoder."):
            decoder.append(param)
        else:
            encoder.append(param)
    lr = cfg.train.lr
    groups = []
    if memory:
        groups.append({"name": "memory", "params": memory,
                       "lr": lr * cfg.train.memory_lr_scale})
    if decoder:
        groups.append({"name": "decoder", "params": decoder,
                       "lr": lr * cfg.train.decoder_lr_scale})
    if encoder:
        groups.append({"name": "encoder", "params": encoder,
                       "lr": lr * cfg.train.encoder_lr_scale})
    if inert:
        log.info("excluded from the optimizer (no reachable gradient): %s",
                 ", ".join(inert))
    return groups
