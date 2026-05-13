"""Offside determination and rendering module."""
from __future__ import annotations

import cv2
import numpy as np

# Standard pitch coordinates (in meters) for homography calibration
# Top-left, Top-right, Bottom-right, Bottom-left
PITCH_DST = [
    (0.0, 0.0),
    (105.0, 0.0),
    (105.0, 68.0),
    (0.0, 68.0)
]

# Standard pitch width in meters (matches the Y-coordinates above)
PITCH_WIDTH_Y = 68.0

def get_pitch_bounded_line(H: np.ndarray, pitch_x: float) -> tuple[tuple[int, int], tuple[int, int]]:
    """Returns image coordinates for a line across the pitch at a specific pitch X."""
    H_inv = np.linalg.inv(H)
    
    # Define endpoints at the top touchline (Y=0) and bottom touchline (Y=68)
    pts_pitch = np.array([[[pitch_x, 0.0], [pitch_x, PITCH_WIDTH_Y]]], dtype=np.float32)
    pts_img = cv2.perspectiveTransform(pts_pitch, H_inv)[0]
    
    # Return safely mapped integer coordinates for drawing
    return tuple(map(int, pts_img[0])), tuple(map(int, pts_img[1]))

def snap_to_defender_edge(frame: np.ndarray, click_pt: tuple[float, float], goal_side: str) -> tuple[float, float]:
    """
    AI/CV helper: Snaps the user's click to the defender's rearmost pixel.
    Uses a local bounding box and edge detection to find the player's silhouette.
    """
    cx, cy = int(click_pt[0]), int(click_pt[1])
    h, w = frame.shape[:2]

    # Define a 60x60 Region of Interest (ROI) around the click
    box_size = 60
    x1, y1 = max(0, cx - box_size // 2), max(0, cy - box_size // 2)
    x2, y2 = min(w, cx + box_size // 2), min(h, cy + box_size // 2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return click_pt

    # Simple AI proxy: Grayscale and Canny edge detection to find the player silhouette
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find the coordinates of all edge pixels within our box
    y_coords, x_coords = np.where(edges > 0)
    if len(x_coords) == 0:
        return click_pt

    # Determine the rearmost pixel based on the direction of play
    if goal_side == "left":
        # Defending left goal -> find the furthest left pixel
        idx = np.argmin(x_coords)
    else:
        # Defending right goal -> find the furthest right pixel
        idx = np.argmax(x_coords)

    # Translate local ROI coordinates back to the global frame
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

    # 1. Draw Manual Line (if any)
    if manual_line:
        cv2.line(frame,
                 (int(manual_line[0][0]), int(manual_line[0][1])),
                 (int(manual_line[1][0]), int(manual_line[1][1])),
                 (255, 255, 0), 2, cv2.LINE_AA)

    # If the pitch hasn't been calibrated yet, we can't project perspective lines
    if H is None:
        return verdict

    # 2. Helper to project points into 2D Pitch Space
    def to_pitch(pt):
        p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        return cv2.perspectiveTransform(p, H)[0][0]

    pitch_att_x = None
    pitch_def_x = None

    if attacker:
        pitch_att_x = to_pitch(attacker)[0]
        # Draw attacker marker
        cv2.circle(frame, (int(attacker[0]), int(attacker[1])), 5, (0, 165, 255), -1)

    if defender:
        # AI SNAP: Adjust the rough click to the rearmost edge of the player
        ai_defender = snap_to_defender_edge(frame, defender, goal_side)
        pitch_def_x = to_pitch(ai_defender)[0]

        # Calculate a line perfectly bounded by the touchlines, parallel to the goal line
        pt1, pt2 = get_pitch_bounded_line(H, pitch_def_x)

        # Draw the AI-snapped line and point
        cv2.line(frame, pt1, pt2, (255, 50, 50), 2, cv2.LINE_AA)
        cv2.circle(frame, (int(ai_defender[0]), int(ai_defender[1])), 5, (255, 50, 50), -1)

    # 3. Determine Verdict based on X-axis overlap in Pitch Space
    if pitch_att_x is not None and pitch_def_x is not None:
        if goal_side == "right":
            verdict = "OFFSIDE" if pitch_att_x > pitch_def_x else "ON-SIDE"
        else:
            verdict = "OFFSIDE" if pitch_att_x < pitch_def_x else "ON-SIDE"

    return verdict
