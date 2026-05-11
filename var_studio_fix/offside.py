"""Offside line drawing + verdict.

Two modes are supported:
  1. Calibrated mode  — uses homography H to draw a true perpendicular-to-goal
     line through the defender (and optionally the attacker), and computes
     the verdict from pitch-space coordinates.
  2. Manual line mode — the user draws a single straight line on the screen
     (two clicks). The attacker's position relative to that line + a chosen
     goal side determines the verdict. No calibration required.
"""
from __future__ import annotations

import cv2
import numpy as np

from .homography import apply_h, invert


PITCH_W = 105.0
PITCH_H = 68.0
PITCH_DST = [(0.0, 0.0), (PITCH_W, 0.0), (PITCH_W, PITCH_H), (0.0, PITCH_H)]


def _clip_line_to_frame(line: tuple[tuple[float, float], tuple[float, float]],
                        frame_h: int, frame_w: int,
                        margin: int = 20) -> tuple[tuple[float, float], tuple[float, float]]:
    """Clip line endpoints to stay within frame + margin, preventing spectator areas.
    
    Keeps line within a reasonable field region by constraining to:
    - Horizontal: [margin, frame_w - margin]
    - Vertical: [margin, frame_h - margin]
    """
    (x1, y1), (x2, y2) = line
    
    # Clamp to frame boundaries with margin
    x1 = max(margin, min(frame_w - margin - 1, x1))
    y1 = max(margin, min(frame_h - margin - 1, y1))
    x2 = max(margin, min(frame_w - margin - 1, x2))
    y2 = max(margin, min(frame_h - margin - 1, y2))
    
    return ((x1, y1), (x2, y2))


def perpendicular_line_through(H: np.ndarray, screen_pt: tuple[float, float]
                               ) -> tuple[tuple[int, int], tuple[int, int]]:
    px = apply_h(H, screen_pt)
    Hinv = invert(H)
    a = apply_h(Hinv, (px[0], 0.0))
    b = apply_h(Hinv, (px[0], PITCH_H))
    return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))


def _side_of_line(line: tuple[tuple[float, float], tuple[float, float]],
                  pt: tuple[float, float]) -> float:
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)


def draw_offside(img: np.ndarray,
                 H: np.ndarray | None,
                 attacker: tuple[float, float] | None,
                 defender: tuple[float, float] | None,
                 manual_line: tuple[tuple[float, float],
                                    tuple[float, float]] | None = None,
                 goal_side: str = "right") -> str | None:
    """Draw markers / lines and return verdict string ('ONSIDE'/'OFFSIDE') or None.
    
    For manual lines: clips line to stay within frame to avoid drawing into spectator areas.
    """
    verdict: str | None = None
    frame_h, frame_w = img.shape[:2]

    # --- markers ---------------------------------------------------------
    if attacker is not None:
        cv2.circle(img, (int(attacker[0]), int(attacker[1])), 7,
                   (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "A", (int(attacker[0]) + 9, int(attacker[1]) - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    if defender is not None:
        cv2.circle(img, (int(defender[0]), int(defender[1])), 7,
                   (255, 90, 90), -1, cv2.LINE_AA)
        cv2.putText(img, "D", (int(defender[0]) + 9, int(defender[1]) - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 90, 90), 2, cv2.LINE_AA)

    # --- offside line + verdict -----------------------------------------
    line_screen: tuple[tuple[float, float], tuple[float, float]] | None = None

    if manual_line is not None:
        # Clip manual line to frame to prevent drawing into spectator areas
        line_screen = _clip_line_to_frame(manual_line, frame_h, frame_w, margin=30)
        cv2.line(img,
                 (int(line_screen[0][0]), int(line_screen[0][1])),
                 (int(line_screen[1][0]), int(line_screen[1][1])),
                 (255, 80, 80), 2, cv2.LINE_AA)
    elif H is not None and defender is not None:
        d1, d2 = perpendicular_line_through(H, defender)
        line_screen = ((float(d1[0]), float(d1[1])),
                       (float(d2[0]), float(d2[1])))
        cv2.line(img, d1, d2, (255, 80, 80), 2, cv2.LINE_AA)
        if attacker is not None:
            a1, a2 = perpendicular_line_through(H, attacker)
            cv2.line(img, a1, a2, (0, 220, 255), 1, cv2.LINE_AA)

    if line_screen is not None and attacker is not None:
        if H is not None and manual_line is None:
            ax = apply_h(H, attacker)[0]
            dx = apply_h(H, defender)[0] if defender is not None else ax
            attacker_ahead = ax > dx if goal_side == "right" else ax < dx
        else:
            # use drawn line + reference point (defender or right edge)
            side_attacker = _side_of_line(line_screen, attacker)
            ref_pt = defender if defender is not None else (
                (img.shape[1] - 1.0, img.shape[0] / 2.0)
                if goal_side == "right"
                else (0.0, img.shape[0] / 2.0))
            side_goal = _side_of_line(line_screen, ref_pt)
            attacker_ahead = (side_attacker * side_goal) > 0 and \
                             abs(side_attacker) > 1e-3
        verdict = "OFFSIDE" if attacker_ahead else "ONSIDE"
        color = (0, 0, 255) if verdict == "OFFSIDE" else (0, 200, 0)
        cv2.rectangle(img, (10, 14), (260, 60), (0, 0, 0), -1)
        cv2.putText(img, verdict, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)

    return verdict
