"""Auto pitch calibration — find 4 strong line-intersection corners.

Strategy (fast, dependency-light):
  1. Downsample to ~480p luminance
  2. Mask roughly green (pitch) regions — keep white lines on grass
  3. Canny edges -> HoughLinesP
  4. Cluster lines into ~horizontal vs ~vertical
  5. Pick top 2 of each, intersect to get 4 corners
  6. Order TL, TR, BR, BL and return in original-frame coordinates
"""
from __future__ import annotations

import cv2
import numpy as np


def _intersect(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def detect_pitch_corners(frame_bgr: np.ndarray) -> list[tuple[float, float]] | None:
    H, W = frame_bgr.shape[:2]
    scale = 480.0 / H
    small = cv2.resize(frame_bgr, (int(W * scale), 480))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 30, 30), (95, 255, 255))
    green = cv2.dilate(green, np.ones((9, 9), np.uint8))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=green)
    _, white = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(white, 60, 180)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                            minLineLength=80, maxLineGap=20)
    if lines is None or len(lines) < 4:
        return None

    horiz, vert = [], []
    for l in lines[:, 0]:
        x1, y1, x2, y2 = l
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) < 25 or abs(abs(ang) - 180) < 25:
            horiz.append((l, abs(x2 - x1)))
        elif abs(abs(ang) - 90) < 25:
            vert.append((l, abs(y2 - y1)))

    if len(horiz) < 2 or len(vert) < 2:
        return None

    horiz.sort(key=lambda x: -x[1]); vert.sort(key=lambda x: -x[1])
    h1, h2 = horiz[0][0], horiz[1][0]
    v1, v2 = vert[0][0], vert[1][0]

    corners = []
    for hl in (h1, h2):
        for vl in (v1, v2):
            p = _intersect(hl, vl)
            if p is not None:
                corners.append(p)
    if len(corners) != 4:
        return None

    # order TL, TR, BR, BL
    corners = sorted(corners, key=lambda p: p[1])
    top = sorted(corners[:2], key=lambda p: p[0])
    bot = sorted(corners[2:], key=lambda p: p[0])
    ordered = [top[0], top[1], bot[1], bot[0]]
    inv = 1.0 / scale
    return [(p[0] * inv, p[1] * inv) for p in ordered]
