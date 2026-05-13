"""Offside determination and rendering module."""
from __future__ import annotations

import cv2
import numpy as np

# Standard pitch coordinates (in meters) for homography calibration
PITCH_DST = [
    (0.0, 0.0),
    (105.0, 0.0),
    (105.0, 68.0),
    (0.0, 68.0)
]

# The standard width of a football pitch in meters
PITCH_WIDTH_Y = 68.0

def get_pitch_bounded_line(H: np.ndarray, pitch_x: float) -> tuple[tuple[int, int], tuple[int, int]]:
    """Mathematically bounds the line strictly between the top (Y=0) and bottom (Y=68) touchlines."""
    H_inv = np.linalg.inv(H)
    
    # We restrict the line entirely to the 0m -> 68m width of the calibrated pitch
    pts_pitch = np.array([[[pitch_x, 0.0], [pitch_x, PITCH_WIDTH_Y]]], dtype=np.float32)
    pts_img = cv2.perspectiveTransform(pts_pitch, H_inv)[0]
    
    return tuple(map(int, pts_img[0])), tuple(map(int, pts_img[1]))

def snap_to_defender_edge(frame: np.ndarray, click_pt: tuple[float, float], goal_side: str) -> tuple[float, float]:
    """AI/CV helper: Snaps the user's click to the defender's rearmost pixel."""
    cx, cy = int(click_pt[0]), int(click_pt[1])
    h, w = frame.shape[:2]

    box_size = 60
    x1, y1 = max(0, cx - box_size // 2), max(0, cy - box_size // 2)
    x2, y2 = min(w, cx + box_size // 2), min(h, cy + box_size // 2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return click_pt

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    y_coords, x_coords = np.where(edges > 0)
    if len(x_coords) == 0:
        return click_pt

    if goal_side == "left":
        idx = np.argmin(x_coords)
    else:
        idx = np.argmax(x_coords)

    snapped_x = x1 + x_coords[idx]
    snapped_y = y1 + y_coords[idx]

    return float(snapped_x), float(snapped_y)

def draw_offside(
    frame: np.ndarray,
    H: np.ndarray | None,
    attacker: tuple[float, float] | None = None,
    defender: tuple[float, float] | None = None,
    manual_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
    goal_side: str = "right"
) -> str | None:
    """Draws offside lines and returns the verdict."""
    verdict = None

    if manual_line:
        cv2.line(frame,
                 (int(manual_line[0][0]), int(manual_line[0][1])),
                 (int(manual_line[1][0]), int(manual_line[1][1])),
                 (255, 255, 0), 2, cv2.LINE_AA)

    if H is None:
        return verdict

    def to_pitch(pt):
        p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        return cv2.perspectiveTransform(p, H)[0][0]

    pitch_att_x = None
    pitch_def_x = None

    if attacker:
        pitch_att_x = to_pitch(attacker)[0]
        cv2.drawMarker(frame, (int(attacker[0]), int(attacker[1])), (0, 165, 255), cv2.MARKER_CROSS, 10, 2)

    if defender:
        ai_defender = snap_to_defender_edge(frame, defender, goal_side)
        pitch_def_x = to_pitch(ai_defender)[0]

        # 1. Get the STRICTLY bounded line coordinates based on Homography math
        pt1, pt2 = get_pitch_bounded_line(H, pitch_def_x)

        # 2. Draw it natively. It will stop exactly where the calibrated pitch stops.
        cv2.line(frame, pt1, pt2, (255, 50, 50), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, (int(ai_defender[0]), int(ai_defender[1])), (255, 50, 50), cv2.MARKER_CROSS, 10, 2)

    if pitch_att_x is not None and pitch_def_x is not None:
        if goal_side == "right":
            verdict = "OFFSIDE" if pitch_att_x > pitch_def_x else "ON-SIDE"
        else:
            verdict = "OFFSIDE" if pitch_att_x < pitch_def_x else "ON-SIDE"

    return verdict
