"""Pixel conditioning applied before SAM 3 sees a frame.

CLAHE is always-on at inference (`smokeseg.config.preprocess_clahe`), so it is
always-on here. It runs on the **full frame** and the crop is taken afterwards,
matching the poc's ordering — CLAHE is a local-contrast operator, so running it
on a small crop produces different pixels than running it on the frame.
"""

from __future__ import annotations

import cv2
import numpy as np


def clahe_rgb(image_rgb: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel of LAB, preserving color."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    op = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    l_chan = op.apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2RGB)


def load_frame(path: str, clahe: bool, clip: float, grid: int) -> np.ndarray:
    """Read a frame as conditioned RGB uint8 (H, W, 3)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return clahe_rgb(rgb, clip, grid) if clahe else rgb


def decode_frame(data: bytes, clahe: bool, clip: float, grid: int) -> np.ndarray:
    """Same as `load_frame`, for an image that arrived as bytes rather than a path.

    The serving endpoint takes an upload, so there is no path to hand `imread`.
    Kept as its own function instead of routing `load_frame` through it: the
    DataLoader's frame reader is on the hot path of every recorded number, and it
    does not need to change for an upload to work. The one step that has to be
    shared is `clahe_rgb` on the *full* frame before any crop — see the module
    docstring — and it is.
    """
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("not a decodable image (expected JPEG/PNG bytes)")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return clahe_rgb(rgb, clip, grid) if clahe else rgb
