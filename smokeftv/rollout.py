"""The clip loop, and the one line that is the whole experiment.

`prompted_frames_are_conditioning: false` reduces to the `key = ...` assignment
below: frame 0 and explicit re-inits go to `cond_frame_outputs`, and EVERY other
frame goes to `non_cond_frame_outputs` regardless of whether it carried a
prompt. Upstream gets this the other way round —
`sam3_tracking_predictor.py:264-266` is
`is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond`, with
that flag hardcoded True at `:54` — and under per-frame prompting that makes
EVERY frame a conditioning frame: the cond cap saturates, the recency FIFO is
evicted, and the module runs far outside its pretrained regime.

The temporal encoding follows for free. `sam3_tracker_base.py:673-678` forces
`t = 0` for selected cond frames, so all of them index the same
`maskmem_tpos_enc` slot; a mid-clip frame misfiled as conditioning loses its
temporal identity entirely.
"""

from __future__ import annotations

import torch

from smokeftv.track_step import TrackOut, box_to_point_inputs, track_step_train

TOKENS_PER_MEM_FRAME = 72 * 72      # the stride-16 grid the bank stores
TOKENS_PER_OBJ_PTR = 4              # C // mem_dim = 256 // 64


class BankProbe:
    """What memory attention actually attended, read off the tensor it got.

    A forward pre-hook on `transformer.encoder`, so it cannot disagree with the
    model the way a reconstruction from `output_dict` could. Pair its rows with
    the rollout's own (n_cond, n_recent) labels and assert they match — that
    assert is what catches a silent reclassification, which is otherwise
    invisible until the metrics move three days later.
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._handle = None

    def attach(self, tracker):
        self._handle = tracker.transformer.encoder.register_forward_pre_hook(
            self._hook, with_kwargs=True)
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, args, kwargs):
        prompt = kwargs.get("prompt")
        n_ptr = int(kwargs.get("num_obj_ptr_tokens") or 0)
        if prompt is None:
            return None
        n_tok = int(prompt.shape[0])
        self.rows.append({
            "mem_tokens": n_tok - n_ptr,
            "mem_frames": (n_tok - n_ptr) // TOKENS_PER_MEM_FRAME,
            "ptr_tokens": n_ptr,
            "ptr_frames": n_ptr // TOKENS_PER_OBJ_PTR,
        })
        return None


def _step(tracker, batch, cfg, t: int, length: int, output_dict: dict,
          with_box: bool, gt_mask) -> TrackOut:
    feats, pos, sizes = batch["feats"][t]
    return track_step_train(
        tracker,
        frame_idx=t,
        is_init_cond_frame=(t == 0),
        current_vision_feats=feats,
        current_vision_pos_embeds=pos,
        feat_sizes=sizes,
        point_inputs=(box_to_point_inputs(batch["boxes"][t], cfg.crop.image_size)
                      if with_box else None),
        output_dict=output_dict,
        num_frames=length,
        multimask_output=(cfg.model.multimask_mode == "always" or not with_box),
        gt_mask=gt_mask,
        mem_mask_source=cfg.model.mem_mask_source,
    )


def rollout_clip(tracker, batch, cfg, *, keep_mask, probe: BankProbe | None = None,
                 reprompt=None) -> tuple[list[TrackOut], list[dict]]:
    """Run one clip end to end, returning a `TrackOut` per frame.

    `batch["feats"]` is a list of length L, each a 3-level
    `(current_vision_feats, current_vision_pos_embeds, feat_sizes)` triple.
    Frame indices handed to `track_step_train` are LOCAL (0..L-1); see its
    docstring for why that is load-bearing rather than tidy.

    `batch["targets_high"]` may be None. The GT eval passes None, because
    `mem_mask_source: supervised_best` picks the memory candidate from the GT
    mask whenever one is given — in eval mode too — which is a leak at test time.

    `reprompt(t, out) -> bool` is the eval's `conditional` arm: an unprompted
    frame is tracked from memory first, and re-run WITH its box when the
    callback rejects the propagated mask. Only the accepted output is filed, so
    memory never sees the rejected attempt.
    """
    length = len(batch["feats"])
    output_dict: dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    outs: list[TrackOut] = []
    labels: list[dict] = []
    truncate = cfg.model.memory.truncate_bptt
    targets_high = batch.get("targets_high")

    for t in range(length):
        is_init = (t == 0)
        prompted = bool(keep_mask[t])
        gt_mask = targets_high[t] if targets_high is not None else None

        out = _step(tracker, batch, cfg, t, length, output_dict, prompted, gt_mask)
        reprompted = False
        if not prompted and reprompt is not None and reprompt(t, out):
            out = _step(tracker, batch, cfg, t, length, output_dict, True, gt_mask)
            prompted = reprompted = True
        outs.append(out)

        # ---- THE override. is_cond is (t == 0 or a re-init) ONLY ---------- #
        key = "cond_frame_outputs" if is_init else "non_cond_frame_outputs"
        entry = {
            "maskmem_features": out.maskmem_features,
            "maskmem_pos_enc": out.maskmem_pos_enc,
            "obj_ptr": out.obj_ptr,
            "pred_masks": out.low_res_masks,
            "object_score_logits": out.object_score_logits,
        }
        if truncate is not None and (length - 1 - t) > truncate:
            # Truncated BPTT: detach memory older than `truncate` frames so the
            # graph stops growing. Changes what the memory encoder learns, so it
            # is recorded in the run config rather than being a quiet default.
            entry = {k: (v.detach() if torch.is_tensor(v) else
                         [x.detach() for x in v] if isinstance(v, list) else v)
                     for k, v in entry.items()}
        output_dict[key][t] = entry

        labels.append({"t": t, "is_cond": is_init, "prompted": prompted,
                       "reprompted": reprompted,
                       "n_cond": len(output_dict["cond_frame_outputs"]),
                       "n_recent": min(len(output_dict["non_cond_frame_outputs"]),
                                       tracker.num_maskmem - 1)})
    return outs, labels
