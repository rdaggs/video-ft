"""The run config: dataclasses, `extends` resolution, overrides, validation.

Two invariants, both deliberate and both worth keeping:

* **Nothing here reads the environment.** A run's geometry is what its config
  says, not what the shell exported.
* **Exactly one filesystem read**, `splits.train_incidents: auto`. Everything
  else is pure, which is what keeps the no-GPU preflight a second.

The module is also **torch-free**. `resolve_dtype` imports torch lazily for the
same reason.

Precedence is **snapshot < config file < CLI override**, and a list is a leaf:
a config's `val_incidents` *replaces* the snapshot's rather than appending,
because appending would make "hold out exactly these cameras" inexpressible.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from smokeftv.boxes import BboxRepairParams, BoundaryExtendParams, EventGroupingParams
from smokeftv.clips import ClipParams
from smokeftv.incidents import AUTO, auto_train_incidents, camera_of
from smokeftv.masklets import MaskletParams

REPO_ROOT = Path(__file__).resolve().parent.parent
SAM3_REPO_ROOT = REPO_ROOT / "sam3"
SNAPSHOT_NAME = "training_config.yaml"
SNAPSHOT_PATH = REPO_ROOT / SNAPSHOT_NAME

# Restated rather than imported from the modules they describe, so config
# validation never imports the thing it is validating.
SELECTABLE_METRICS = ("iou_fused", "iou_polygon", "iou_best", "iou_selected")
SCHEDULES = ("none", "cosine")
CROP_MODES = ("full_frame", "track_window", "per_frame_box")
WINDOW_SCOPES = ("masklet", "clip")
BOX_SOURCES = ("annotations", "detector", "mixed")
INFERENCE_MODES = ("prompt_every_frame", "keyframe", "conditional")
CLIP_POLICIES = ("sequential", "shuffle")
MEM_MASK_SOURCES = ("supervised_best", "predicted_best", "gt")
MULTIMASK_MODES = ("always", "sam3_native")

# Name prefixes of the tracker's memory path, for the freeze policy. The five
# bare Parameters need exact match, not `startswith`, or a future
# `no_mem_embed_v2` would be caught by the `no_mem_embed` rule.
MEMORY_MODULES: dict[str, tuple[str, ...]] = {
    "attention": ("transformer.",),
    "encoder": ("maskmem_backbone.",),
    "pointers": ("obj_ptr_proj.", "obj_ptr_tpos_proj."),
    "mask_down": ("mask_downsample.",),
}
MEMORY_TOKENS = ("maskmem_tpos_enc", "no_mem_embed", "no_mem_pos_enc",
                 "no_obj_ptr", "no_obj_embed_spatial")


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class DataConfig:
    root: str = "/home/rileydaggs/data/cvf-2026"
    train_incidents: list[str] | str = field(default_factory=list)
    val_incidents: list[str] = field(default_factory=list)
    test_incidents: list[str] = field(default_factory=list)
    gt_dataset_root: str = "/home/rileydaggs/data/segmentation-test-set-new"
    gt_merge: str = "union"
    min_mask_pixels: int = 16
    clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    masklets: MaskletParams = field(default_factory=MaskletParams)
    clips: ClipParams = field(default_factory=ClipParams)


@dataclass
class CropConfig:
    mode: str = "track_window"
    window_scope: str = "masklet"
    image_size: int = 1008
    union_pad: float = 0.35
    bbox_pad: float | None = 0.5
    square_infer: bool = True
    crop_pad_frac: float = 0.5
    crop_pad_top_extra: float = 0.5


@dataclass
class BoxJitterParams:
    """Per-frame box padding, p ~ U(pad_min, pad_max): each side grows by p/2 of
    the box's OWN extent on that axis, so the box becomes (1+p)w x (1+p)h about
    the same centre. Per-axis on purpose, unlike `crops.pad_box`'s longer-side
    margin, which would turn a wide box nearly square at p=2.

    Deliberately a run knob, not a snapshot key: it is an augmentation, and the
    GT eval forces it off (`evaluate.eval_overrides`) so every run is scored on
    the same boxes. Masklets are still linked from the clean boxes, so the
    sample set does not move; only the crop windows and the prompts see it.
    """
    enabled: bool = False
    pad_min: float = 0.5
    pad_max: float = 2.0
    seed: int = 0


@dataclass
class PromptConfig:
    box_source: str = "annotations"
    detector_boxes_path: str | None = None
    mixed_detector_frac: float = 0.5
    bbox_propagate_frames: int = -1
    bbox_from_polygon: bool = True
    bbox_from_polygon_pad: float = 0.5
    bbox_event_grouping: EventGroupingParams = field(default_factory=EventGroupingParams)
    bbox_boundary_extend: BoundaryExtendParams = field(default_factory=BoundaryExtendParams)
    bbox_repair: BboxRepairParams = field(default_factory=BboxRepairParams)
    box_jitter: BoxJitterParams = field(default_factory=BoxJitterParams)


@dataclass
class MemoryConfig:
    max_recent_frames: int = 6
    max_cond_frames_in_attn: int = 2
    prompted_frames_are_conditioning: bool = False
    keep_first_cond_frame: bool = True
    temporal_pos_encoding: bool = True
    object_pointers: bool = True
    log_bank_composition: bool = True
    # Which named slices of MEMORY_MODULES the optimizer holds. Empty means the
    # memory path is frozen, which is only sensible for a control arm.
    train: list[str] = field(default_factory=list)
    # Truncated BPTT. null keeps the whole clip in one graph; an int detaches
    # memory older than that many frames. Changes what the memory encoder
    # learns, so it belongs in the run record.
    truncate_bptt: int | None = None

    @property
    def num_maskmem(self) -> int:
        """SAM 3's name for this. The FIFO loop is `range(1, num_maskmem)`, so
        7 gives 6 recency slots."""
        return int(self.max_recent_frames) + 1


@dataclass
class ModelConfig:
    sam_version: str = "sam3"
    checkpoint: str | None = None
    train_mask_decoder: bool = False
    train_prompt_encoder: bool = False
    multimask: bool = True
    # `always` forces multimask_output=True on every frame. Under SAM 3's stock
    # multimask_min_pt_num=0 / max_pt_num=1 a box is two points, so
    # `_use_multimask` returns False on prompted frames and True on dropout
    # frames — 1 candidate against 3, with different object-pointer token
    # semantics between them. `sam3_native` reproduces that; it is an ablation.
    multimask_mode: str = "always"
    # `.train()` turns on 0.1 dropout across all 4 memory-attention layers
    # (16 nn.Dropout + 8 RoPEAttention). The single-clip overfit test cannot
    # reach loss < 0.01 with it on.
    memory_attn_dropout: float = 0.0
    # Substitute object-presence from the GT rather than the frozen head. When
    # `object_score_logits <= 0` the mask becomes a constant -1024 with exactly
    # zero gradient and the memory for that frame is poisoned — a silent stall
    # that reads as "the loss plateaued".
    force_obj_appearing: bool = True
    # Which of the multimask candidates is handed to the memory encoder. Stock
    # uses the IoU head's pick, which disagrees with the candidate the loss
    # supervises while that head is frozen and uncalibrated on smoke.
    mem_mask_source: str = "supervised_best"
    grad_checkpoint_memory_attn: bool = False
    memory: MemoryConfig = field(default_factory=MemoryConfig)


@dataclass
class LossConfig:
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    iou_head_weight: float = 1.0
    presence_weight: float = 0.0
    smokebase_weight: float = 0.0
    smokebase_radius_px: int = 5
    temporal_consistency_weight: float = 0.0
    # Per-frame weighting inside a clip. Frame 0 is conditioning and trivially
    # easy; dropout frames are the only ones measuring whether memory works.
    cond_frame_weight: float = 1.0
    prompted_frame_weight: float = 1.0
    dropout_frame_weight: float = 1.0


@dataclass
class DropoutSchedule:
    """Per clip, draw a keep-rate p ~ U(p_min, p_max); per frame, Bernoulli(p).
    Frame 0 is always kept.

    At U(0.3, 0.9): 35% of frames unprompted, mean longest unprompted run 1.95,
    p90 4 — so memory carries the object for up to 4 consecutive frames in 10%
    of clips, while 9% of clips stay fully prompted and still show the model the
    inference regime.
    """

    p_min: float = 1.0
    p_max: float = 1.0
    warmup_epochs: int = 0


@dataclass
class FeatureCacheConfig:
    enabled: bool = False
    root: str = "/data3/rileydaggs/sam3-video-cvf/cache"


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 1
    grad_accum: int = 1
    lr: float = 1.0e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    amp_dtype: str = "bfloat16"
    num_workers: int = 8
    clip_policy: str = "shuffle"
    loss_size: int = 1008
    log_every: int = 50
    eval_every: int = 1
    eval_polygon: bool = True
    select_metric: str = "iou_polygon"
    inference_mode: str = "conditional"
    lr_schedule: str = "cosine"
    lr_warmup_frac: float = 0.05
    lr_min_factor: float = 0.0
    memory_lr_scale: float = 1.0
    decoder_lr_scale: float = 0.1
    encoder_lr_scale: float = 0.01
    init_ckpt: list[str] | str | None = None
    resume: str | None = None
    save_resume: bool = True
    loss: LossConfig = field(default_factory=LossConfig)
    dropout: DropoutSchedule = field(default_factory=DropoutSchedule)


@dataclass
class RunConfig:
    logs_root: str = "logs"
    ckpts_root: str = "ckpts"
    run_name_from: str = "stdout"
    require_clean_git: bool = True
    save_clip_manifest: bool = True
    save_prompt_schedule: bool = True
    keep_checkpoints: list[str] = field(default_factory=lambda: ["best", "last"])
    feature_cache_root: str = "/data3/rileydaggs/sam3-video-cvf/cache"


@dataclass
class Config:
    run_name: str = "unnamed"
    seed: int = 0
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)
    feature_cache: FeatureCacheConfig = field(default_factory=FeatureCacheConfig)

    @property
    def run_dir(self) -> Path:
        root = Path(self.run.ckpts_root)
        return (root if root.is_absolute() else REPO_ROOT / root) / self.run_name

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

_PARAM_CLASSES = {
    MaskletParams: "masklets", ClipParams: "clips",
    EventGroupingParams: "bbox_event_grouping",
    BoundaryExtendParams: "bbox_boundary_extend",
    BboxRepairParams: "bbox_repair",
}


def _field_default(f):
    """A field's default instance, or MISSING. `dataclasses.MISSING` is a
    sentinel *type*, not None, so `is not None` is the wrong test and produces
    a confusing TypeError three frames away."""
    if f.default_factory is not MISSING:        # type: ignore[misc]
        return f.default_factory()              # type: ignore[misc]
    return f.default


def _from_dict(cls, data: Mapping[str, Any]):
    """Recursive dataclass builder. **Unknown keys raise.**

    A typo'd `train.learning_rate` that silently does nothing is a wasted
    GPU-hour and a confusing plot.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"{cls.__name__}: expected a mapping; got {data!r}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown config keys {unknown}. "
            f"Known keys are {sorted(known)}.")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        default = _field_default(f)
        if type(default) in _PARAM_CLASSES:
            kwargs[f.name] = type(default).coerce(value, _PARAM_CLASSES[type(default)])
        elif is_dataclass(default) and isinstance(value, Mapping):
            kwargs[f.name] = _from_dict(type(default), value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict:
    """Recursive dict merge. Neither input is mutated.

    **A list is a leaf.** `val_incidents` in a config replaces the snapshot's
    rather than appending, because appending would let you only ever add to a
    held-out set and never state it.
    """
    out = dict(base)
    for key, value in over.items():
        if (key in out and isinstance(out[key], Mapping)
                and isinstance(value, Mapping)):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_extends(raw: dict, config_dir: Path) -> dict:
    """Resolve `extends:`, which accepts only the one snapshot.

    Not a general include: a config inheriting through a chain of files is
    exactly how it stops being readable on its own.
    """
    name = raw.pop("extends", None)
    if name is None:
        return raw
    if Path(name).name != SNAPSHOT_NAME:
        raise ValueError(
            f"extends: {name!r} — only {SNAPSHOT_NAME!r} may be extended. It is "
            "a hook for the cross-consumer snapshot, not a general include, "
            "because a config inheriting through a chain of files is exactly "
            "how it stops being readable on its own.")
    from smokeftv import config_all  # here, to break the import cycle
    candidate = (config_dir / name).resolve()
    path = candidate if candidate.is_file() else config_all.SNAPSHOT_PATH
    return _deep_merge(config_all.train_base(config_all.load(path)), raw)


def _apply_override(data: dict, dotted: str) -> None:
    """`a.b.c=value`, with the RHS parsed by `yaml.safe_load`.

    So `3e-5` is a float, `true`/`false` are bools, `[a,b]` is a list, `null` is
    None, and a bare word is a string. Applied to the raw dict before
    construction, so the dataclasses still validate it.
    """
    key, _, raw = dotted.partition("=")
    if not raw:
        raise ValueError(f"Override {dotted!r} is not of the form key=value")
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(raw)


# --------------------------------------------------------------------------- #
# Validators — each names the remedy, because a config error should not need
# a reading of this file to act on.
# --------------------------------------------------------------------------- #

def _resolve_incidents(data: DataConfig) -> None:
    if isinstance(data.val_incidents, str) or isinstance(data.test_incidents, str):
        raise ValueError(
            "splits.val_incidents / test_incidents must be lists of names. The "
            "held-out split is a deliberate choice of cameras, not whatever is "
            "left over — name them.")
    if isinstance(data.train_incidents, str):
        if data.train_incidents != AUTO:
            raise ValueError(
                f"data.train_incidents must be a list or {AUTO!r}; "
                f"got {data.train_incidents!r}")
        data.train_incidents = auto_train_incidents(
            data.root, data.val_incidents, data.test_incidents)


def _check_leakage(data: DataConfig) -> None:
    held = {camera_of(n) for n in data.val_incidents}
    held |= {camera_of(n) for n in data.test_incidents}
    leaked = sorted({camera_of(n) for n in data.train_incidents} & held)
    if leaked:
        raise ValueError(
            f"train and held-out splits share cameras {leaked}. Splits hold out "
            "whole cameras, not whole incidents: two incidents on one camera "
            "share a horizon and usually an overlapping field of view, so this "
            "is not a held-out measurement.")


def _check_crop(crop: CropConfig) -> None:
    if crop.mode not in CROP_MODES:
        raise ValueError(f"crop.mode must be one of {list(CROP_MODES)}; got {crop.mode!r}")
    if crop.window_scope not in WINDOW_SCOPES:
        raise ValueError(f"crop.window_scope must be one of {list(WINDOW_SCOPES)}; "
                         f"got {crop.window_scope!r}")
    if crop.mode == "track_window" and crop.union_pad < 0:
        raise ValueError(f"crop.union_pad must be >= 0; got {crop.union_pad!r}")
    if crop.square_infer and crop.bbox_pad is None and crop.mode == "per_frame_box":
        raise ValueError(
            "crop.square_infer is true but crop.bbox_pad is null, so "
            "crops.crop_for_box takes the legacy compute_crop branch and the "
            "window is never squared. Set bbox_pad, or square_infer: false.")


def _check_prompt(prompt: PromptConfig) -> None:
    jitter = prompt.box_jitter
    if not 0.0 <= jitter.pad_min <= jitter.pad_max:
        raise ValueError(f"prompt.box_jitter needs 0 <= pad_min <= pad_max; got "
                         f"{jitter.pad_min!r}, {jitter.pad_max!r}")
    if prompt.box_source not in BOX_SOURCES:
        raise ValueError(f"prompting.box_source must be one of {list(BOX_SOURCES)}; "
                         f"got {prompt.box_source!r}")
    if prompt.box_source != "annotations":
        raise NotImplementedError(
            f"prompting.box_source: {prompt.box_source!r} is not implemented. "
            "Every one of the 529 annotations.json files carries "
            "box_source: labelbox_*_groundtruth and no detector dump exists on "
            "disk; sam3_pred.coco.json is not a substitute because its info "
            "block says prompt_mode: pvs, i.e. SAM 3 prompted by the GT box. "
            "See smokeftv.prompting.DETECTOR_DUMP_SPEC for the format a dump "
            "would need, and divergence `detector_prompts`.")
    if prompt.detector_boxes_path is not None:
        raise ValueError(
            "prompting.detector_boxes_path is set but box_source is "
            "'annotations', so nothing reads it. A path nothing reads is worse "
            "than null.")
    if prompt.bbox_from_polygon and prompt.bbox_from_polygon_pad <= 0:
        raise ValueError(
            "prompting.bbox_from_polygon_pad must be > 0 when bbox_from_polygon "
            "is on: a tight bbox around the answer is not a prompt any detector "
            "produces, and the padding is what stops it being one.")


def _check_memory(model: ModelConfig) -> None:
    mem = model.memory
    unknown = sorted(set(mem.train) - set(MEMORY_MODULES))
    if unknown:
        raise ValueError(f"model.memory.train names unknown slices {unknown}; "
                         f"known are {sorted(MEMORY_MODULES)}")
    if mem.max_cond_frames_in_attn < 1:
        raise ValueError("model.memory.max_cond_frames_in_attn must be >= 1")
    if mem.max_recent_frames < 1:
        raise ValueError("model.memory.max_recent_frames must be >= 1")
    for name in ("temporal_pos_encoding", "object_pointers"):
        if not getattr(mem, name):
            raise NotImplementedError(
                f"model.memory.{name}: false has no switch on Sam3TrackerBase. "
                "obj_ptr_proj, obj_ptr_tpos_proj and maskmem_tpos_enc are built "
                "unconditionally; use_obj_ptrs_in_encoder / "
                "add_tpos_enc_to_obj_ptrs exist only on the sam3.1 multiplex "
                "path and shared.sam_version is sam3. These two are assertions "
                "about the vendored source, not knobs — --check verifies them.")
    if model.multimask_mode not in MULTIMASK_MODES:
        raise ValueError(f"model.multimask_mode must be one of {list(MULTIMASK_MODES)}")
    if model.mem_mask_source not in MEM_MASK_SOURCES:
        raise ValueError(f"model.mem_mask_source must be one of {list(MEM_MASK_SOURCES)}")
    if mem.truncate_bptt is not None and mem.truncate_bptt < 1:
        raise ValueError("model.memory.truncate_bptt must be >= 1 or null")


def _check_train(train: TrainConfig) -> None:
    if train.select_metric not in SELECTABLE_METRICS:
        raise ValueError(f"train.select_metric must be one of "
                         f"{list(SELECTABLE_METRICS)}; got {train.select_metric!r}")
    if train.select_metric == "iou_polygon" and not train.eval_polygon:
        raise ValueError(
            "train.select_metric is 'iou_polygon' but train.eval_polygon is "
            "false, so the metric selecting best.pt is never computed.")
    if train.lr_schedule not in SCHEDULES:
        raise ValueError(f"train.lr_schedule must be one of {list(SCHEDULES)}")
    if train.clip_policy not in CLIP_POLICIES:
        raise ValueError(f"train.clip_policy must be one of {list(CLIP_POLICIES)}")
    if train.inference_mode not in INFERENCE_MODES:
        raise ValueError(f"train.inference_mode must be one of {list(INFERENCE_MODES)}")
    if train.resume and train.init_ckpt:
        raise ValueError(
            "train.resume and train.init_ckpt cannot both be set. `resume` "
            "continues a run under its own name with its optimizer; `init_ckpt` "
            "starts a NEW run from existing weights with a fresh one.")
    if train.grad_accum < 1:
        raise ValueError("train.grad_accum must be >= 1")
    if not 0.0 <= train.lr_warmup_frac < 1.0:
        raise ValueError("train.lr_warmup_frac must be in [0, 1)")
    d = train.dropout
    if not 0.0 <= d.p_min <= d.p_max <= 1.0:
        raise ValueError(
            f"train.dropout must satisfy 0 <= p_min <= p_max <= 1; got "
            f"p_min={d.p_min!r} p_max={d.p_max!r}. p is a KEEP rate: 1.0 means "
            "every frame keeps its box.")
    if d.warmup_epochs >= train.epochs:
        raise ValueError(
            f"train.dropout.warmup_epochs ({d.warmup_epochs}) must be < "
            f"train.epochs ({train.epochs}), or the schedule never starts.")


def _check_cache(cfg: "Config") -> None:
    """The feature cache is only valid when the encoder output is a pure
    function of (masklet, frame). Assert that, rather than silently serving
    stale features."""
    if not cfg.feature_cache.enabled:
        return
    if cfg.crop.mode == "per_frame_box":
        raise ValueError(
            "feature_cache.enabled with crop.mode: per_frame_box. The window "
            "then depends on the per-frame box, so the encoder output is not a "
            "function of (masklet, frame) and the cache key does not identify "
            "it. Set feature_cache.enabled: false to run that ablation.")
    if cfg.crop.window_scope == "clip":
        raise ValueError(
            "feature_cache.enabled with crop.window_scope: clip. Clip "
            "membership moves with clips.stride_jitter and the clip start, so a "
            "per-clip window changes every epoch seed and a cache entry would "
            "be valid for exactly one epoch.")
    if cfg.prompt.bbox_boundary_extend.source == "prediction":
        raise ValueError(
            "feature_cache.enabled with bbox_boundary_extend.source: "
            "prediction. The window would depend on model weights that change "
            "every step; there is no key that stays valid.")
    if cfg.model.train_mask_decoder:
        raise ValueError(
            "feature_cache.enabled with model.train_mask_decoder: true. "
            "sam_mask_decoder.conv_s0 and conv_s1 are applied at ENCODE time "
            "(sam3_tracker_base.py:432-437) and are baked into the cache, so "
            "unfreezing the decoder makes every cached tensor stale after step "
            "1. Exclude conv_s0/conv_s1 from the unfreeze set, or disable the "
            "cache.")


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "poc_field_map" in raw or "divergences" in raw:
        raise ValueError(
            f"{path} looks like the snapshot itself, not a run config. The "
            "snapshot is not loadable as a Config — read it with "
            "`python -m smokeftv.config_all --show`.")
    raw = _resolve_extends(raw, Path(path).resolve().parent)
    for override in overrides or []:
        _apply_override(raw, override)
    cfg = _from_dict(Config, raw)
    _resolve_incidents(cfg.data)
    _check_leakage(cfg.data)
    _check_crop(cfg.crop)
    _check_prompt(cfg.prompt)
    _check_memory(cfg.model)
    _check_train(cfg.train)
    _check_cache(cfg)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Write the fully literal record. `extends` is gone by construction —
    `Config` has no such field — so this file's values cannot move when the
    snapshot is next edited. It is the file you reach for to reproduce a run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")


def resolve_dtype(name: str):
    import torch
    return {"float32": torch.float32, "bfloat16": torch.bfloat16,
            "float16": torch.float16}[name]
