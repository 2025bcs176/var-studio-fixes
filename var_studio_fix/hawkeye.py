"""Hawk-Eye top-down pitch view.

Renders a 2D bird's-eye football pitch and projects calibrated points
(attacker, defender, ball, offside lines) onto it via the homography H
(screen -> pitch coordinates in metres).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .homography import apply_h

# Standard pitch dimensions in meters defined locally
PITCH_W = 105.0
PITCH_H = 68.0

# Cached static pitch background (grass + lines). Drawn once per size.
_PITCH_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _draw_pitch(w: int, h: int) -> np.ndarray:
    # Clean solid professional green background (no striped shading)
    img = np.full((h, w, 3), (38, 110, 52), dtype=np.uint8)

    pad = int(min(w, h) * 0.04)

    def s(x_m: float, y_m: float) -> tuple[int, int]:
        x = pad + int(x_m / PITCH_W * (w - 2 * pad))
        y = pad + int(y_m / PITCH_H * (h - 2 * pad))
        return x, y

    white = (240, 240, 240)
    th = max(1, w // 360)
    
    # outer boundary
    cv2.rectangle(img, s(0, 0), s(PITCH_W, PITCH_H), white, th)
    # halfway line
    cv2.line(img, s(PITCH_W / 2, 0), s(PITCH_W / 2, PITCH_H), white, th)
    # center circle
    cv2.circle(img, s(PITCH_W / 2, PITCH_H / 2),
               int(9.15 / PITCH_W * (w - 2 * pad)), white, th)
    cv2.circle(img, s(PITCH_W / 2, PITCH_H / 2), max(2, th + 1), white, -1)
    
    # penalty boxes (16.5m x 40.3m), 6-yard (5.5m x 18.32m)
    for sign in (0, 1):
        x0 = 0 if sign == 0 else PITCH_W - 16.5
        x1 = 16.5 if sign == 0 else PITCH_W
        cv2.rectangle(img, s(x0, (PITCH_H - 40.3) / 2),
                      s(x1, (PITCH_H + 40.3) / 2), white, th)
        
        x0b = 0 if sign == 0 else PITCH_W - 5.5
        x1b = 5.5 if sign == 0 else PITCH_W
        cv2.rectangle(img, s(x0b, (PITCH_H - 18.32) / 2),
                      s(x1b, (PITCH_H + 18.32) / 2), white, th)
        
        # penalty spot 11m
        spot = 11.0 if sign == 0 else PITCH_W - 11.0
        cv2.circle(img, s(spot, PITCH_H / 2), max(2, th + 1), white, -1)
        
    return img


def render_hawkeye(view_w: int, view_h: int, H: Optional[np.ndarray],
                   attacker: Optional[tuple[float, float]],
                   defender: Optional[tuple[float, float]],
                   trail: Optional[list[tuple[float, float]]] = None
                   ) -> np.ndarray:
    """Compose the top-down pitch with overlays."""
    key = (view_w, view_h)
    base = _PITCH_CACHE.get(key)
    if base is None:
        base = _draw_pitch(view_w, view_h)
        _PITCH_CACHE[key] = base
    img = base.copy()

    pad = int(min(view_w, view_h) * 0.04)

    def to_view(px: float, py: float) -> tuple[int, int]:
        x = pad + int(np.clip(px / PITCH_W, 0, 1) * (view_w - 2 * pad))
        y = pad + int(np.clip(py / PITCH_H, 0, 1) * (view_h - 2 * pad))
        return x, y

    if H is None:
        cv2.putText(img, "AI Calibrating...",
                    (20, view_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (230, 230, 230), 1, cv2.LINE_AA)
        return img

    def proj(p):
        m = apply_h(H, p)
        return float(m[0]), float(m[1])

    # offside lines (vertical in pitch space)
    if attacker is not None:
        ax, _ = proj(attacker)
        cv2.line(img, to_view(ax, 0), to_view(ax, PITCH_H),
                 (0, 220, 255), 2, cv2.LINE_AA)
    if defender is not None:
        dx, _ = proj(defender)
        cv2.line(img, to_view(dx, 0), to_view(dx, PITCH_H),
                 (255, 80, 80), 2, cv2.LINE_AA)

    # players
    if attacker is not None:
        ax, ay = proj(attacker)
        cv2.circle(img, to_view(ax, ay), 7, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(img, to_view(ax, ay), 8, (20, 20, 20), 1, cv2.LINE_AA)
    if defender is not None:
        dx, dy = proj(defender)
        cv2.circle(img, to_view(dx, dy), 7, (255, 80, 80), -1, cv2.LINE_AA)
        cv2.circle(img, to_view(dx, dy), 8, (20, 20, 20), 1, cv2.LINE_AA)

    # trail (ball / motion)
    if trail:
        prev = None
        for i, pt in enumerate(trail[-30:]):
            mx, my = proj(pt)
            v = to_view(mx, my)
            alpha = (i + 1) / max(1, len(trail[-30:]))
            cv2.circle(img, v, 3, (255, 255, 255), -1, cv2.LINE_AA)
            if prev is not None:
                cv2.line(img, prev, v,
                         (int(255 * alpha), int(255 * alpha), 255), 1, cv2.LINE_AA)
            prev = v

    # verdict
    if attacker is not None and defender is not None:
        ax, _ = proj(attacker); dx, _ = proj(defender)
        verdict = "OFFSIDE" if ax > dx else "ONSIDE"
        color = (0, 0, 255) if verdict == "OFFSIDE" else (60, 220, 90)
        cv2.putText(img, verdict, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)
    return img
