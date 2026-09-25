"""A differentiable `track_step`, forked op-for-op from the vendored tracker.

`sam3_tracker_base.track_step` (`:929-1102`) cannot be called directly for
training, for two reasons that are both about what it throws away:

* **It discards the multimask candidates.** It unpacks `_forward_sam_heads`'s
  seven-tuple and keeps only the *selected* mask, so the image repo's loss rule
  — supervise the highest hard-IoU of three, MSE the IoU head on all three — is
  not expressible on its return value.
* **It drops `object_score_logits` in train mode** (`:1018-1021`), which is the
  one signal that says whether the presence head has silently zeroed a frame's
  gradient.

So this is the same sequence of operations, returning everything. It calls the
*vendored* `_prepare_memory_conditioned_features` and `_forward_sam_heads`
unchanged — all the memory semantics we care about live in those two, and a
second copy of them would drift.

What this file deliberately does NOT do is go through `Sam3TrackerPredictor`.
Every entry point there is decorated `@torch.inference_mode()`, so its tensors
can never enter autograd; `propagate_in_video_preflight` re-files outputs by
storage key; and `clear_non_cond_mem_around_input=True` deletes the recency
memory around every prompted frame, which is the exact opposite of what this
experiment is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrackOut:
    """Everything a clip loss and a memory bank need from one frame."""

    low_res_multimasks: torch.Tensor    # (B, M, 288, 288) — the loss reads this
    high_res_multimasks: torch.Tensor   # (B, M, 1008, 1008) fp32
    ious: torch.Tensor                  # (B, M) — IoU-head prediction
    low_res_masks: torch.Tensor         # (B, 1, 288, 288) the selected candidate
    high_res_masks: torch.Tensor        # (B, 1, 1008, 1008)
    obj_ptr: torch.Tensor               # (B, 256)
    object_score_logits: torch.Tensor   # (B, 1) — ALWAYS, unlike the vendored one
    maskmem_features: torch.Tensor | None
    maskmem_pos_enc: list | None
    mem_index: torch.Tensor             # (B,) which candidate went into memory


def box_to_point_inputs(boxes: torch.Tensor, image_size: int) -> dict:
    """A box becomes its two corners, labelled 2 and 3 — SAM 2's convention and
    what `add_new_points_or_box` does (`sam3_tracking_predictor.py:224-232`).

    `boxes` is (B, 4) in LTRB pixels already scaled to `image_size`.
    """
    coords = boxes.reshape(-1, 2, 2).float()
    labels = torch.full((coords.shape[0], 2), 2, dtype=torch.int32,
                        device=coords.device)
    labels[:, 1] = 3
    return {"point_coords": coords, "point_labels": labels}


def track_step_train(
    tracker,
    *,
    frame_idx: int,
    is_init_cond_frame: bool,
    current_vision_feats: list[torch.Tensor],
    current_vision_pos_embeds: list[torch.Tensor],
    feat_sizes: list[tuple[int, int]],
    point_inputs: dict | None,
    output_dict: dict,
    num_frames: int,
    multimask_output: bool = True,
    gt_mask: torch.Tensor | None = None,
    mem_mask_source: str = "supervised_best",
    run_mem_encoder: bool = True,
) -> TrackOut:
    """One frame. `frame_idx` is the LOCAL clip index, 0..L-1.

    The local index is not cosmetic: the recency FIFO looks up
    `output_dict["non_cond_frame_outputs"][frame_idx - t_rel]`
    (`sam3_tracker_base.py:625-643`), so a strided set of real frame indices
    would silently produce an EMPTY memory bank with no error at all.
    """
    # High-resolution feature maps for the SAM head, (HW)BC => BCHW.
    if len(current_vision_feats) > 1:
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
        ]
    else:
        high_res_features = None

    # Memory attention. Vendored, unchanged — this is the module being trained.
    pix_feat_with_mem = tracker._prepare_memory_conditioned_features(
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        current_vision_feats=current_vision_feats[-1:],
        current_vision_pos_embeds=current_vision_pos_embeds[-1:],
        feat_sizes=feat_sizes[-1:],
        output_dict=output_dict,
        num_frames=num_frames,
    )

    # `multimask_output` is passed explicitly rather than taken from
    # `_use_multimask`. Under the stock multimask_min_pt_num=0 /
    # max_pt_num=1 a box is two points, so that helper returns False on every
    # box-prompted frame and True on every dropout frame — one candidate
    # against three, with different object-pointer token semantics between
    # them, inside a single clip.
    #
    # `gt_masks` reaches `_forward_sam_heads:333`, which uses it for
    # `is_obj_appearing` when `tracker.teacher_force_obj_scores_for_mem` is on.
    # That matters more than it looks: when `object_score_logits <= 0` the mask
    # is replaced by a constant NO_OBJ_SCORE and the frame contributes EXACTLY
    # ZERO gradient, which reads as "the loss plateaued".
    sam_outputs = tracker._forward_sam_heads(
        backbone_features=pix_feat_with_mem,
        point_inputs=point_inputs,
        mask_inputs=None,
        high_res_features=high_res_features,
        multimask_output=multimask_output,
        gt_masks=gt_mask,
    )
    (low_res_multimasks, high_res_multimasks, ious,
     low_res_masks, high_res_masks, obj_ptr, object_score_logits) = sam_outputs

    # Which candidate is written into memory. The vendored path always uses the
    # IoU head's pick; while that head is frozen and uncalibrated on smoke it
    # disagrees with the candidate the loss actually supervises, and the
    # disagreement is absorbed silently.
    mem_index = ious.argmax(dim=-1)
    if mem_mask_source == "supervised_best" and gt_mask is not None:
        with torch.no_grad():
            from smokeftv.losses import hard_iou
            b, m = high_res_multimasks.shape[:2]
            hw = high_res_multimasks.shape[-2:]
            target = gt_mask.reshape(b, 1, *hw).expand(b, m, *hw).reshape(b * m, *hw)
            real = hard_iou(high_res_multimasks.reshape(b * m, *hw), target).view(b, m)
            mem_index = real.argmax(dim=-1)
    rows = torch.arange(high_res_multimasks.shape[0], device=high_res_multimasks.device)
    mem_mask = high_res_multimasks[rows, mem_index].unsqueeze(1)
    if mem_mask_source == "gt" and gt_mask is not None:
        mem_mask = (gt_mask.reshape(mem_mask.shape).float() * 20.0) - 10.0

    maskmem_features = maskmem_pos_enc = None
    if run_mem_encoder and tracker.num_maskmem > 0:
        maskmem_features, maskmem_pos_enc = tracker._encode_new_memory(
            image=None,                     # unused for SimpleMaskEncoder
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=mem_mask,
            object_score_logits=object_score_logits,
            is_mask_from_pts=(point_inputs is not None),
            output_dict=output_dict,
            is_init_cond_frame=is_init_cond_frame,
        )

    return TrackOut(
        low_res_multimasks=low_res_multimasks,
        high_res_multimasks=high_res_multimasks,
        ious=ious,
        low_res_masks=low_res_masks,
        high_res_masks=high_res_masks,
        obj_ptr=obj_ptr,
        object_score_logits=object_score_logits,
        maskmem_features=maskmem_features,
        maskmem_pos_enc=maskmem_pos_enc,
        mem_index=mem_index,
    )
