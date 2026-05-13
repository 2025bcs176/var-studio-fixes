"""Real-time image processing: blur, sharpen, brightness/contrast, zoom/pan.

All ops use OpenCV (SIMD/threaded) so 1080p stays ~real-time on CPU.
Frame quality can be enhanced using Lanczos interpolation and Detail Enhancement.
"""
from __future__ import annotations

import cv2
import numpy as np

from .store import Settings


# Interpolation quality levels (higher = slower but better quality)
INTERPOLATION_QUALITY = {
    0: cv2.INTER_NEAREST,      # Fastest, lowest quality (for previews)
    1: cv2.INTER_LINEAR,       # Bilinear (3x3 neighborhood)
    2: cv2.INTER_CUBIC,        # Bicubic (4x4 neighborhood)
    3: cv2.INTER_LANCZOS4,     # Best for upscaling/zooming (8x8 neighborhood)
}


def apply_pipeline(frame_bgr: np.ndarray, s: Settings) -> np.ndarray:
    """Apply blur -> brightness/contrast -> zoom/pan -> sharpen -> enhance."""
    img = frame_bgr
    
    # 1. Gaussian blur (applied before zoom to smooth source noise)
    if s.blur > 0.05:
        k = max(3, int(s.blur * 4) | 1)  # odd kernel
        img = cv2.GaussianBlur(img, (k, k), s.blur)

    # 2. Brightness / contrast
    if abs(s.contrast - 1.0) > 1e-3 or abs(s.brightness) > 1e-3:
        img = cv2.convertScaleAbs(img, alpha=s.contrast, beta=s.brightness * 255.0)

    h, w = img.shape[:2]

    # Target output dimensions based on output_scale multiplier
    scale = getattr(s, "output_scale", 1.0)
    if scale > 1.001:
        tw, th = int(w * scale), int(h * scale)
    else:
        tw, th = w, h

    # Default to 3 (Lanczos4) to ensure high resolution preservation during zoom
    quality = getattr(s, "output_quality", 3)
    quality = max(0, min(3, int(quality)))
    interp = INTERPOLATION_QUALITY[quality]

    # 3. Zoom & pan: crop the ROI then resize to target
    if s.zoom > 1.001:
        cw, ch = int(w / s.zoom), int(h / s.zoom)
        cx = int(w / 2 + s.pan_x * (w - cw) / 2)
        cy = int(h / 2 + s.pan_y * (h - ch) / 2)
        x0 = max(0, min(w - cw, cx - cw // 2))
        y0 = max(0, min(h - ch, cy - ch // 2))
        
        crop = img[y0:y0 + ch, x0:x0 + cw]
        # Resizing the crop back up using Lanczos interpolation
        img = cv2.resize(crop, (tw, th), interpolation=interp)
    elif tw != w or th != h:
        img = cv2.resize(img, (tw, th), interpolation=interp)

    # 4. Unsharp mask sharpen (Crucial: Apply AFTER zoom to keep upscaled edges crisp!)
    if s.sharpen > 0.01:
        blur = cv2.GaussianBlur(img, (0, 0), 1.5)
        img = cv2.addWeighted(img, 1 + s.sharpen, blur, -s.sharpen, 0)

    # 5. Advanced Detail Enhancement (Increases perceived picture quality)
    # Checks if 'enhance_details' exists in Settings and is True
    if getattr(s, "enhance_details", False):
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

    return img


def to_luminance(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
