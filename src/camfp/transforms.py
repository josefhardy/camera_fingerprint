"""Image transformations used to test robustness, e.g. 'does it survive WhatsApp?'.

All functions take and return uint8 RGB arrays of the same size, so a transformed photo
can be compared directly with the camera's fingerprint.

Note: `whatsapp_like` is only an approximation (downscale + JPEG). For real social media
copies, use the VISION dataset's WhatsApp and Facebook versions.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def jpeg(rgb: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def rescale(rgb: np.ndarray, scale: float) -> np.ndarray:
    """Downscale then upscale back: keeps pixel alignment but destroys fine detail."""
    h, w = rgb.shape[:2]
    small = Image.fromarray(rgb).resize((max(1, int(w * scale)), max(1, int(h * scale))),
                                        Image.BICUBIC)
    return np.asarray(small.resize((w, h), Image.BICUBIC))


def blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    out = gaussian_filter(rgb.astype(np.float64), sigma=(sigma, sigma, 0))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def shift(rgb: np.ndarray, pixels: int) -> np.ndarray:
    """Shift by a few pixels (like a slight crop). Breaks alignment with the fingerprint."""
    return np.roll(rgb, (pixels, pixels), axis=(0, 1))


def whatsapp_like(rgb: np.ndarray, scale: float = 0.6, quality: int = 70) -> np.ndarray:
    return jpeg(rescale(rgb, scale), quality)


def parse(spec: str):
    """Turn 'jpeg:70', 'rescale:0.5', 'blur:1', 'shift:2', 'whatsapp' or 'none' into a function."""
    name, _, arg = spec.partition(":")
    funcs = {"jpeg": lambda x: jpeg(x, int(arg)),
             "rescale": lambda x: rescale(x, float(arg)),
             "blur": lambda x: blur(x, float(arg)),
             "shift": lambda x: shift(x, int(arg)),
             "whatsapp": whatsapp_like,
             "none": lambda x: x}
    if name not in funcs:
        raise ValueError(f"Unknown transform {spec!r}; options: {sorted(funcs)}")
    return funcs[name]
