"""FFT helpers — 2D magnitude/phase spectra."""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 2D — image spectra
# ---------------------------------------------------------------------------
def fft2_spectra(gray: np.ndarray, size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Return (log-magnitude, phase) shifted to DC-center, both uint8 viz buffers
    of shape (size, size). Default size=128 keeps the panel real-time.
    """
    img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    img -= img.mean()
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(F))
    pha = np.angle(F)
    mag = (mag / (mag.max() + 1e-9) * 255).astype(np.uint8)
    pha = ((pha + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    mag = cv2.applyColorMap(mag, cv2.COLORMAP_VIRIDIS)
    pha = cv2.applyColorMap(pha, cv2.COLORMAP_TWILIGHT)
    return mag, pha
