"""Threaded video decoder with frame cache.

Uses OpenCV for video frames (works for local files and HTTP MP4 URLs).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np


class LRU(OrderedDict):
    def __init__(self, cap: int = 64):
        super().__init__()
        self.cap = cap

    def put(self, k, v):
        if k in self:
            self.move_to_end(k)
        self[k] = v
        if len(self) > self.cap:
            self.popitem(last=False)

    def get(self, k, default=None):
        if k in self:
            self.move_to_end(k)
            return self[k]
        return default


class VideoEngine:
    def __init__(self) -> None:
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.n_frames: int = 0
        self.width = 0
        self.height = 0
        self.cache = LRU(128)
        self._lock = threading.Lock()
        self._path: str = ""
        self._last_idx: int = -10

    @property
    def loaded(self) -> bool:
        return self.cap is not None

    def open(self, path: str) -> None:
        with self._lock:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                self.cap = None
                raise RuntimeError(f"Cannot open video: {path}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.cache.clear()
            self._path = path

    @property
    def duration(self) -> float:
        return self.n_frames / max(1.0, self.fps)

    def read_frame(self, idx: int) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        idx = max(0, min(self.n_frames - 1, idx)) if self.n_frames else max(0, idx)
        cached = self.cache.get(idx)
        if cached is not None:
            return cached
        with self._lock:
            if idx != self._last_idx + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok:
                return None
            self._last_idx = idx
            self.cache.put(idx, frame)
            return frame

    def read_at(self, t_seconds: float) -> Optional[np.ndarray]:
        return self.read_frame(int(round(t_seconds * self.fps)))

    def release(self) -> None:
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
