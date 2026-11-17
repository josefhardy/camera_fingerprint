"""Simulated cameras, for testing the pipeline without downloading anything.

Each simulated camera has the two kinds of trace this project studies:

* a **device fingerprint** (PRNU): every pixel's sensitivity differs slightly, by around
  1%, in a fixed random pattern unique to that individual sensor;
* **model-level processing**: cameras of the same model share colour gains and a
  sharpening/smoothing filter, mimicking each manufacturer's image pipeline.

Real cameras are far messier, so results on synthetic data only show that the code works,
not how well the methods do in practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter


@dataclass
class SyntheticCamera:
    brand: str
    model: str
    owner: str
    prnu: np.ndarray          # (H, W) multiplicative sensitivity pattern
    gains: np.ndarray         # (3,) colour gains shared by the model
    kernel: np.ndarray        # (3, 3) processing filter shared by the model
    read_noise: float

    @property
    def folder_name(self) -> str:
        return f"{self.brand}_{self.model}_{self.owner}"


def make_cameras(n_brands: int = 2, models_per_brand: int = 2, devices_per_model: int = 2,
                 size: int = 256, prnu_strength: float = 0.01, seed: int = 0) -> list[SyntheticCamera]:
    rng = np.random.default_rng(seed)
    cameras = []
    for b in range(n_brands):
        for m in range(models_per_brand):
            gains = 1 + rng.normal(0, 0.04, 3)
            sharp = rng.uniform(-0.15, 0.15)
            kernel = np.full((3, 3), -sharp / 8)
            kernel[1, 1] = 1 + sharp
            for d in range(devices_per_model):
                cameras.append(SyntheticCamera(
                    brand=f"Brand{b}", model=f"Model{b}{m}", owner=f"unit{d}",
                    prnu=rng.normal(0, prnu_strength, (size, size)),
                    gains=gains, kernel=kernel, read_noise=rng.uniform(1.0, 2.0),
                ))
    return cameras


def random_scene(size: int, rng: np.random.Generator, flat: bool = False) -> np.ndarray:
    """A plausible-looking RGB scene in 0-255 (float)."""
    if flat:
        base = rng.uniform(140, 200)
        yy, xx = np.mgrid[0:size, 0:size] / size
        gray = base + 20 * (yy - 0.5) * rng.normal() + 10 * (xx - 0.5) * rng.normal()
        return np.repeat(gray[..., None], 3, axis=2)
    scene = np.zeros((size, size, 3))
    for sigma, amp in [(size / 4, 60), (size / 16, 30), (2, 12)]:
        noise = gaussian_filter(rng.normal(0, 1, (size, size, 3)), sigma=(sigma, sigma, 0))
        scene += amp * noise / (noise.std() + 1e-9)
    for _ in range(rng.integers(2, 6)):  # a few hard-edged objects
        y, x = rng.integers(0, size, 2)
        h, w = rng.integers(size // 10, size // 3, 2)
        scene[y : y + h, x : x + w] += rng.normal(0, 40, 3)
    return np.clip(scene + 120, 5, 245)


def capture(camera: SyntheticCamera, scene: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate the camera taking a photo of `scene`: returns uint8 RGB."""
    img = scene * (1 + camera.prnu[..., None]) * camera.gains
    img = np.stack([convolve(img[..., c], camera.kernel, mode="reflect") for c in range(3)], axis=2)
    img += rng.normal(0, camera.read_noise, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def write_dataset(root: str | Path, cameras: list[SyntheticCamera], n_natural: int = 30,
                  n_flat: int = 10, seed: int = 0, jpeg_quality: int | None = None) -> Path:
    """Write images in the 'own photos' layout so `camfp index --own` can read them."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    size = cameras[0].prnu.shape[0]
    for cam in cameras:
        folder = root / cam.folder_name
        (folder / "flat").mkdir(parents=True, exist_ok=True)
        for kind, n in [("natural", n_natural), ("flat", n_flat)]:
            for i in range(n):
                img = capture(cam, random_scene(size, rng, flat=kind == "flat"), rng)
                out_dir = folder / "flat" if kind == "flat" else folder
                if jpeg_quality:
                    Image.fromarray(img).save(out_dir / f"{kind}_{i:04d}.jpg", quality=jpeg_quality)
                else:
                    Image.fromarray(img).save(out_dir / f"{kind}_{i:04d}.png")
    return root
