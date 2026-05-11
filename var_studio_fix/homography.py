"""4-point homography (DLT) + projection helpers."""
from __future__ import annotations

import numpy as np

Pt = tuple[float, float]


def homography(src: list[Pt], dst: list[Pt]) -> np.ndarray:
    """Solve H (3x3) such that dst ~ H * src for 4 point correspondences."""
    assert len(src) == 4 and len(dst) == 4
    A = []
    b = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7], 1.0]])
    return H


def apply_h(H: np.ndarray, p: Pt) -> Pt:
    v = H @ np.array([p[0], p[1], 1.0])
    return (float(v[0] / v[2]), float(v[1] / v[2]))


def invert(H: np.ndarray) -> np.ndarray:
    return np.linalg.inv(H)
