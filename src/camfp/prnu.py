"""Classic sensor fingerprinting with PRNU (Photo-Response Non-Uniformity).

Every pixel on a camera sensor responds to light slightly differently because of tiny
manufacturing flaws. A photo I is roughly  I = I0 * (1 + K) + noise,  where K is the
sensor's fixed pattern: its fingerprint. The method (Lukas, Fridrich & Goljan, 2006; Chen et
al., 2008):

1. Denoise each photo and keep the noise residual  W = I - denoise(I).
2. Estimate the fingerprint from many residuals:  K = sum(W * I) / sum(I^2).
3. To test a new photo, correlate its residual with I * K. A strong, sharp correlation
   peak means the same sensor took it. Peak strength is measured with PCE
   (Peak-to-Correlation Energy); a common decision threshold is PCE > 60.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pywt
from scipy.ndimage import uniform_filter

PCE_THRESHOLD = 60.0


def wavelet_denoise(img: np.ndarray, sigma: float = 3.0, wavelet: str = "db8",
                    levels: int = 4) -> np.ndarray:
    """Mihcak-style wavelet denoiser used in PRNU research.

    For each detail coefficient, estimate the local signal variance over several window
    sizes, keep the smallest estimate, and shrink the coefficient Wiener-style.
    """
    levels = max(1, min(levels, pywt.dwtn_max_level(img.shape, wavelet)))  # small images
    coeffs = pywt.wavedec2(img, wavelet, level=levels, mode="periodization")
    noise_var = sigma**2
    out = [coeffs[0]]
    for detail in coeffs[1:]:
        filtered = []
        for band in detail:
            local = np.full(band.shape, np.inf)
            for w in (3, 5, 7, 9):
                local = np.minimum(local, np.maximum(uniform_filter(band**2, w) - noise_var, 0))
            filtered.append(band * local / (local + noise_var))
        out.append(tuple(filtered))
    den = pywt.waverec2(out, wavelet, mode="periodization")
    return den[: img.shape[0], : img.shape[1]]


def zero_mean(x: np.ndarray) -> np.ndarray:
    """Remove row and column means: kills linear patterns shared by many cameras
    (e.g. from sensor read-out), which would otherwise cause false matches."""
    x = x - x.mean()
    x = x - x.mean(axis=1, keepdims=True)
    return x - x.mean(axis=0, keepdims=True)


def residual(gray: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    return zero_mean(gray - wavelet_denoise(gray, sigma))


def estimate_fingerprint(grays: Iterable[np.ndarray], sigma: float = 3.0,
                         saturation: float = 250.0) -> np.ndarray:
    """Maximum-likelihood fingerprint estimate from several images of one camera.
    Saturated pixels carry no PRNU information, so they are ignored."""
    num = den = None
    for g in grays:
        w = residual(g, sigma)
        mask = g < saturation
        wi, ii = w * g * mask, (g**2) * mask
        num = wi if num is None else num + wi
        den = ii if den is None else den + ii
    if num is None:
        raise ValueError("No images given")
    return zero_mean(num / (den + 1e-8))


def pce(a: np.ndarray, b: np.ndarray, exclude: int = 5) -> float:
    """Signed Peak-to-Correlation Energy at zero shift.

    Circular cross-correlation via FFT; the peak's energy is compared with the average
    energy of the rest of the correlation surface (excluding a small square around the peak).
    """
    a = a - a.mean()
    b = b - b.mean()
    xc = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    peak = xc[0, 0]
    shifted = np.fft.fftshift(xc)
    cy, cx = np.array(xc.shape) // 2
    mask = np.ones_like(shifted, dtype=bool)
    mask[cy - exclude : cy + exclude + 1, cx - exclude : cx + exclude + 1] = False
    energy = np.mean(shifted[mask] ** 2)
    return float(np.sign(peak) * peak**2 / (energy + 1e-20))


def match_score(gray: np.ndarray, fingerprint: np.ndarray, sigma: float = 3.0) -> float:
    """PCE between a photo's residual and the expected fingerprint signal I * K."""
    return pce(residual(gray, sigma), gray * fingerprint)


class PRNUIdentifier:
    """Builds one fingerprint per device and identifies photos by highest PCE."""

    def __init__(self, sigma: float = 3.0):
        self.sigma = sigma
        self.fingerprints: dict[str, np.ndarray] = {}

    def add_device(self, device: str, grays: Iterable[np.ndarray]) -> None:
        self.fingerprints[device] = estimate_fingerprint(grays, self.sigma)

    def scores(self, gray: np.ndarray) -> dict[str, float]:
        w = residual(gray, self.sigma)
        return {dev: pce(w, gray * k) for dev, k in self.fingerprints.items()}

    def identify(self, gray: np.ndarray) -> tuple[str, float, dict[str, float]]:
        scores = self.scores(gray)
        best = max(scores, key=scores.get)
        return best, scores[best], scores
