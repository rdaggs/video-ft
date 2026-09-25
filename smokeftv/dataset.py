"""Clips -> tensors. CLAHE on the full frame, then the crop, then the resize."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from smokeftv.crops import box_into_crop
from smokeftv.boxes import resolve_box
from smokeftv.incidents import frame_path, load_polygons
from smokeftv.masks import polygons_to_mask
from smokeftv.preprocess import load_frame
from smokeftv.prompting import boxes_for_source, jitter_boxes

# Sam3Processor's own chain: uint8 -> square resize -> float -> normalize(0.5,0.5).
# The resize does NOT preserve aspect, which is why crops.compute_window squares
# the window and why masks.polygons_to_mask scales each axis independently.
_MEAN = 0.5
_STD = 0.5


def _to_tensor(rgb: np.ndarray, size: int) -> torch.Tensor:
    import cv2
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return (tensor - _MEAN) / _STD


class ClipDataset(Dataset):
    """One item per clip. Loads L frames, crops them identically, rasterises
    the targets into the same window."""

    def __init__(self, clips, cfg, split: str = "train"):
        self.clips = list(clips)
        self.cfg = cfg
        self.split = split
        self._polygons: dict[str, dict] = {}
        self._boxes: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.clips)

    def _incident_data(self, incident: str, image_hw):
        if incident not in self._polygons:
            polygons, _ = load_polygons(self.cfg.data.root, incident)
            self._polygons[incident] = polygons
            self._boxes[incident] = jitter_boxes(
                boxes_for_source(self.cfg.prompt, self.cfg.data.root, incident),
                self.cfg.prompt.box_jitter, incident, image_hw)
        return self._polygons[incident], self._boxes[incident]

    def __getitem__(self, index: int) -> dict:
        clip = self.clips[index]
        polygons, boxes_by_frame = self._incident_data(clip.incident, clip.image_hw)
        box_frames = sorted(boxes_by_frame)
        size = self.cfg.crop.image_size
        loss_size = self.cfg.train.loss_size
        crop = clip.window

        images, boxes, targets, gaps = [], [], [], []
        for frame in clip.frame_indices:
            rgb = load_frame(str(frame_path(self.cfg.data.root, clip.incident, frame)),
                             self.cfg.data.clahe, self.cfg.data.clahe_clip,
                             self.cfg.data.clahe_grid)
            cropped = rgb[crop[1]:crop[3], crop[0]:crop[2]]
            images.append(_to_tensor(cropped, size))

            resolved = resolve_box(frame, boxes_by_frame,
                                   self.cfg.prompt.bbox_propagate_frames, box_frames)
            polys = polygons.get(frame, [])
            if resolved is not None:
                box, gap = resolved[0][0], resolved[1]
            else:
                from smokeftv.boxes import POLYGON_BOX_GAP, bbox_of_polygons
                from smokeftv.crops import pad_box
                tight = bbox_of_polygons(polys)
                box = pad_box(tight, clip.image_hw,
                              self.cfg.prompt.bbox_from_polygon_pad) if tight else crop
                gap = POLYGON_BOX_GAP
            gaps.append(gap)

            # Crop-local, then normalized by the CROP's own size and rescaled to
            # image_size — the same three steps SAM2Transforms.transform_boxes
            # takes.
            local = box_into_crop(box, crop)
            crop_w, crop_h = max(crop[2] - crop[0], 1), max(crop[3] - crop[1], 1)
            boxes.append(torch.tensor(
                [local[0] / crop_w * size, local[1] / crop_h * size,
                 local[2] / crop_w * size, local[3] / crop_h * size],
                dtype=torch.float32))
            targets.append(torch.from_numpy(
                polygons_to_mask(polys, crop, loss_size)).float())

        return {
            "images": torch.stack(images),            # (L, 3, S, S)
            "boxes": torch.stack(boxes),              # (L, 4)
            "targets": torch.stack(targets),          # (L, loss, loss)
            "box_gap": torch.tensor(gaps),
            "index": index,
            "clip_id": clip.clip_id,
            "incident": clip.incident,
            "crop": torch.tensor(crop),
        }


def collate(items):
    """Batch size is 1 clip; this keeps the clip axis intact."""
    if len(items) != 1:
        raise ValueError("ClipDataset is used at batch_size 1; scale with "
                         "train.grad_accum and ranks instead. Batching clips in "
                         "the model's B dimension would force one dropout "
                         "pattern across the batch, which is the opposite of a "
                         "per-clip keep rate.")
    return items[0]
