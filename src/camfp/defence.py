"""Privacy defences: hide which camera took a photo while keeping it looking the same.

Two approaches, measured by how much they lower the fingerprint match (PCE) and how much
they change the picture (PSNR and SSIM; higher means closer to the original):

* `denoise_defence`        -- strip the sensor noise by denoising, then add fresh random
  noise. Needs nothing in advance, but strong settings soften fine detail.
* `subtract_fingerprint`   -- if you know your own camera's fingerprint K (estimate it from
  some of your own photos), divide it out:  I' = I / (1 + alpha * K). Targeted and
  much gentler on image quality.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .prnu import wavelet_denoise


def denoise_defence(rgb: np.ndarray, strength: float = 6.0, added_noise: float = 1.0,
                    seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.stack([wavelet_denoise(rgb[..., c].astype(np.float64), sigma=strength)
                    for c in range(3)], axis=2)
    out += rng.normal(0, added_noise, out.shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def subtract_fingerprint(rgb: np.ndarray, fingerprint: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    out = rgb.astype(np.float64) / (1 + alpha * fingerprint[..., None])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def image_quality(original: np.ndarray, modified: np.ndarray) -> dict[str, float]:
    if np.array_equal(original, modified):
        return {"psnr": float("inf"), "ssim": 1.0}
    return {
        "psnr": float(peak_signal_noise_ratio(original, modified, data_range=255)),
        "ssim": float(structural_similarity(original, modified, channel_axis=2, data_range=255)),
    }
