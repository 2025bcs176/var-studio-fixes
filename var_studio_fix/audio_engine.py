"""Audio extraction via imageio-ffmpeg + playback via pygame.mixer.

Loads the entire audio track once (mono, 22.05 kHz) into a numpy array,
then serves windowed slices for FFT visualization and plays audio in sync with video.
"""
from __future__ import annotations

import subprocess
import shutil
import threading
from typing import Optional

import numpy as np

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = None

# Fallback to system ffmpeg if imageio_ffmpeg not available
if not FFMPEG or not shutil.which(FFMPEG):
    FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False
    print("Warning: pygame not available. Audio playback will be disabled.")


SR = 22050


class AudioEngine:
    def __init__(self) -> None:
        self.samples: Optional[np.ndarray] = None
        self.sr = SR
        self._sound: Optional[pygame.mixer.Sound] = None
        self._volume = 1.0
        self._filtered: Optional[np.ndarray] = None
        self._filter_ready = threading.Event()  # set when _filtered is valid

    def load(self, path: str) -> None:
        self.samples = None
        self._sound = None
        self._filtered = None
        self._filter_ready.clear()
        try:
            cmd = [FFMPEG, "-nostdin", "-loglevel", "error", "-i", path,
                   "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
            raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                          timeout=30)
            if raw:
                self.samples = np.frombuffer(raw, dtype=np.float32).copy()
        except subprocess.TimeoutExpired:
            print(f"Warning: Audio extraction timeout for {path}")
        except FileNotFoundError:
            print("Warning: ffmpeg not found. Audio will not be available.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: ffmpeg error for {path}: {e}")
        except Exception as e:
            print(f"Warning: Failed to load audio from {path}: {e}")

    def precompute_denoise(self, low_cutoff: float = 0.01,
                           high_cutoff: float = 0.20) -> None:
        """Filter the full audio track in a background thread; signals _filter_ready when done.
        
        This avoids blocking the UI while computing the full FFT.
        """
        if self.samples is None:
            return
        self._filter_ready.clear()
        self._filtered = None

        def _run(samples: np.ndarray, lo: float, hi: float) -> None:
            try:
                F = np.fft.rfft(samples)
                low_bin = max(1, int(lo * len(F)))
                high_bin = max(low_bin + 1, int(hi * len(F)))
                F[:low_bin] = 0
                F[high_bin:] = 0
                self._filtered = np.fft.irfft(F, n=samples.size).astype(np.float32)
            except Exception as e:
                print(f"Warning: denoise precompute failed: {e}")
            finally:
                self._filter_ready.set()

        threading.Thread(target=_run,
                         args=(self.samples.copy(), low_cutoff, high_cutoff),
                         daemon=True).start()

    def play(self, t: float = 0.0, *,
             denoise: bool = False,
             low_cutoff: float = 0.01,
             high_cutoff: float = 0.20) -> None:
        """Start audio playback from time t (in seconds).

        When denoise=True, uses filtered audio if ready (with 0.5s timeout to avoid delay).
        Falls back to raw samples immediately if filter not ready, preventing audio lag.
        """
        if not HAS_PYGAME or self.samples is None:
            return
        try:
            if pygame.mixer.get_busy():
                pygame.mixer.stop()
            
            # Short timeout (0.5s) to prevent audio playback delay
            # If filter isn't ready by then, use raw samples instead
            source = self.samples
            if denoise and self._filtered is not None:
                source = self._filtered
            elif denoise:
                # Non-blocking check: only use filtered if ready
                if self._filter_ready.is_set():
                    source = self._filtered if self._filtered is not None else self.samples
            
            start = int(t * self.sr)
            start = max(0, min(source.size - 1, start))
            chunk = source[start:]
            if chunk.size == 0:
                return
            
            audio_int16 = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
            self._sound = pygame.mixer.Sound(buffer=audio_int16.tobytes())
            self._sound.set_volume(self._volume)
            self._sound.play()
        except Exception as e:
            print(f"Warning: Could not play audio: {e}")

    def stop(self) -> None:
        """Stop audio playback."""
        if not HAS_PYGAME:
            return
        try:
            pygame.mixer.stop()
        except Exception:
            pass

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        if not HAS_PYGAME:
            return False
        try:
            return pygame.mixer.get_busy()
        except Exception:
            return False

    def set_volume(self, vol: float) -> None:
        """Set playback volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, vol))
        if not HAS_PYGAME or self._sound is None:
            return
        try:
            self._sound.set_volume(self._volume)
        except Exception:
            pass

    def window(self, t: float, n: int = 2048) -> Optional[np.ndarray]:
        if self.samples is None or self.samples.size == 0:
            return None
        i = int(t * self.sr)
        i = max(0, min(self.samples.size - n, i))
        return self.samples[i:i + n]
