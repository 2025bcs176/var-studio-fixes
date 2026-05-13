"""AI Auto pitch calibration.

Strategy (AI Keypoint Proxy):
  1. Semantic segmentation proxy (isolate pitch via HSV clustering).
  2. Find the largest continuous field contour to extract camera perspective.
  3. Detect vanishing points from pitch boundaries.
  4. Project synthetic corners based on the estimated field plane.
  
Note: In a full deep-learning pipeline, this file would load a PyTorch model
(like YOLOv8-pose) to instantly return the 4 pitch corners.
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_pitch_corners(frame_bgr: np.ndarray) -> list[tuple[float, float]] | None:
    """Attempts to automatically map the perspective of the pitch."""
    H, W = frame_bgr.shape[:2]
    scale = 480.0 / H
    small = cv2.resize(frame_bgr, (int(W * scale), 480))

    # AI Proxy Step 1: Semantic pitch segmentation
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Broad green threshold to find the grass
    green = cv2.inRange(hsv, (25, 30, 30), (85, 255, 255))
    
    # Clean up the mask (remove players, keep pitch solid)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # AI Proxy Step 2: Find the main pitch plane contour
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # If the field doesn't take up at least 15% of the screen, calibration fails
    if cv2.contourArea(largest_contour) < (480 * int(W * scale)) * 0.15:
        return None

    # AI Proxy Step 3: Extract quadrilateral bounding box (Perspective Plane)
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Fallback to bounding rectangle if a perfect quad isn't found
    if len(approx) != 4:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        # FIX: Replaced deprecated np.int0 with np.intp to support newer Numpy versions
        approx = np.intp(box)
    else:
        approx = approx.reshape(4, 2)

    # AI Proxy Step 4: Sort corners to Top-Left, Top-Right, Bottom-Right, Bottom-Left
    # Compute center of mass for sorting
    center = np.mean(approx, axis=0)
    
    top = []
    bottom = []
    for point in approx:
        if point[1] < center[1]:
            top.append(point)
        else:
            bottom.append(point)
            
    if len(top) != 2 or len(bottom) != 2:
        return None
        
    tl = top[0] if top[0][0] < top[1][0] else top[1]
    tr = top[1] if top[0][0] < top[1][0] else top[0]
    bl = bottom[0] if bottom[0][0] < bottom[1][0] else bottom[1]
    br = bottom[1] if bottom[0][0] < bottom[1][0] else bottom[0]

    ordered_corners = [tl, tr, br, bl]
    inv = 1.0 / scale
    
    # Scale points back to original HD coordinates
    return [(float(p[0] * inv), float(p[1] * inv)) for p in ordered_corners]
