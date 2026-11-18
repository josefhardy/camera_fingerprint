"""Image loading.

Two details matter for camera forensics:

* No EXIF auto-rotation. The sensor's noise pattern is tied to its pixel grid, so images are
  kept exactly as stored, never rotated for display.
* Crops start on even coordinates. Most sensors sit behind a 2x2 colour filter (Bayer)
  pattern, and an odd offset would shift that pattern and blur the camera's traces.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

try:  # iPhone photos (HEIC) need this optional plugin
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


def center_crop(arr: np.ndarray, size: int) -> np.ndarray | None:
    h, w = arr.shape[:2]
    if h < size or w < size:
        return None
    top = ((h - size) // 2) & ~1
    left = ((w - size) // 2) & ~1
    return arr[top : top + size, left : left + size]


def load_rgb(path: str, crop: int | None = 512) -> np.ndarray | None:
    """Load an image as uint8 RGB, centre-cropped to `crop` (None = full image)."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return center_crop(arr, crop) if crop else arr


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Luminance as float64 in 0-255."""
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
