"""Reactive settings store + sorted bookmark list (binary-search)."""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class Settings:
    sharpen: float = 0.0          # 0..2  (unsharp amount)
    blur: float = 0.0             # 0..10 (gaussian sigma)
    brightness: float = 0.0       # -1..1
    contrast: float = 1.0         # 0..3
    zoom: float = 1.0             # 1..8
    pan_x: float = 0.0
    pan_y: float = 0.0
    show_offside: bool = True
    show_overlay: bool = True
    fft_roi: int = 256            # power of two
    harmonics_k: int = 16         # for "clear" reconstruction
    denoise_cutoff: float = 0.10  # normalised freq cutoff (0..0.5) — deprecated, use low/high
    denoise_low_cutoff: float = 0.01   # min freq to keep (normalized, ~500 Hz at 22050 Hz)
    denoise_high_cutoff: float = 0.20  # max freq to keep (normalized, ~4400 Hz at 22050 Hz)
    noise_gate: float = 0.0            # spectral gating (0 = off, 1 = max suppression, 0-12 dB)
    output_scale: float = 1.0          # output resolution multiplier (1.0 = native, 4.0 = 4×)
    volume: float = 1.0               # playback volume (0.0 – 1.0)


class Store:
    """Tiny pub-sub so panels can update without polling."""

    def __init__(self) -> None:
        self.settings = Settings()
        self.bookmarks: List[float] = []  # sorted timestamps (seconds)
        self._listeners: List[Callable[[], None]] = []

    # --- pub/sub -----------------------------------------------------------
    def subscribe(self, fn: Callable[[], None]) -> Callable[[], None]:
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn)

    def notify(self) -> None:
        for fn in list(self._listeners):
            try:
                fn()
            except Exception:
                pass

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self.settings, k, v)
        self.notify()

    # --- bookmarks ---------------------------------------------------------
    def add_bookmark(self, t: float) -> None:
        idx = bisect.bisect_left(self.bookmarks, t)
        if idx < len(self.bookmarks) and abs(self.bookmarks[idx] - t) < 1e-3:
            return
        self.bookmarks.insert(idx, t)
        self.notify()

    def remove_bookmark(self, t: float) -> None:
        idx = bisect.bisect_left(self.bookmarks, t)
        if idx < len(self.bookmarks) and abs(self.bookmarks[idx] - t) < 1e-3:
            self.bookmarks.pop(idx)
            self.notify()

    def nearest_bookmark(self, t: float) -> float | None:
        if not self.bookmarks:
            return None
        idx = bisect.bisect_left(self.bookmarks, t)
        cands = []
        if idx < len(self.bookmarks):
            cands.append(self.bookmarks[idx])
        if idx > 0:
            cands.append(self.bookmarks[idx - 1])
        return min(cands, key=lambda x: abs(x - t))


STORE = Store()
