"""FFT helpers — 2D magnitude/phase spectra and 1D denoise/reconstruct."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 2D — image spectra
# ---------------------------------------------------------------------------
def fft2_spectra(gray: np.ndarray, size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Return (log-magnitude, phase) shifted to DC-center, both uint8 viz buffers
    of shape (size, size). Default size=128 keeps the panel real-time.
    """
    img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    img -= img.mean()
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(F))
    pha = np.angle(F)
    mag = (mag / (mag.max() + 1e-9) * 255).astype(np.uint8)
    pha = ((pha + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    mag = cv2.applyColorMap(mag, cv2.COLORMAP_VIRIDIS)
    pha = cv2.applyColorMap(pha, cv2.COLORMAP_TWILIGHT)
    return mag, pha


# ---------------------------------------------------------------------------
# Spectral Gating (noise floor suppression)
# ---------------------------------------------------------------------------
def estimate_noise_floor(signal: np.ndarray, percentile: int = 25) -> np.ndarray:
    """Estimate noise floor by computing FFT magnitude and taking a low percentile.
    
    Returns per-frequency-bin noise floor estimate.
    """
    x = signal.astype(np.float32) - signal.mean()
    F = np.fft.rfft(x)
    mag = np.abs(F)
    
    # Smooth magnitude spectrum to get noise floor estimate
    # Use low percentile across short windows to avoid speech
    win_size = max(1, len(mag) // 20)
    noise_floor = np.zeros_like(mag)
    for i in range(len(mag)):
        start = max(0, i - win_size // 2)
        end = min(len(mag), i + win_size // 2 + 1)
        noise_floor[i] = np.percentile(mag[start:end], percentile)
    
    return noise_floor


def spectral_gate(signal: np.ndarray, noise_floor: np.ndarray,
                  gate_threshold: float = 3.0) -> np.ndarray:
    """Apply spectral gating: suppress bins below (noise_floor * gate_threshold).
    
    Args:
        signal: Input audio signal
        noise_floor: Pre-computed noise floor per frequency bin
        gate_threshold: dB-like multiplier above noise floor (higher = more suppression)
    
    Returns:
        Gated signal (frequency domain applied, then IFFT)
    """
    if gate_threshold <= 0:
        return signal
    
    x = signal.astype(np.float32) - signal.mean()
    F = np.fft.rfft(x)
    mag = np.abs(F)
    phase = np.angle(F)
    
    # Create gate mask: suppress bins below threshold
    gate_threshold_db = 10 ** (gate_threshold / 20.0)  # convert to linear
    threshold_mag = noise_floor * gate_threshold_db
    mask = np.maximum(mag / (threshold_mag + 1e-10), 0.0)
    mask = np.minimum(mask, 1.0)  # Clamp to [0, 1]
    
    # Apply smooth masking (reduce discontinuities)
    mask = np.minimum(mask, 1.0)  # Ensure [0, 1]
    F_gated = mask * mag * np.exp(1j * phase)
    
    gated = np.fft.irfft(F_gated, n=len(signal))
    return gated.astype(np.float32)


# ---------------------------------------------------------------------------
# 1D — signal triplet (original / denoised / clear)
# ---------------------------------------------------------------------------
@dataclass
class SignalTriplet:
    t: np.ndarray              # time/sample axis
    original: np.ndarray
    denoised: np.ndarray       # high-cut / band-stop
    clear: np.ndarray          # top-K harmonic reconstruction
    freqs: np.ndarray          # rfft freq axis (cycles / sample)
    mag_original: np.ndarray
    mag_denoised: np.ndarray
    mag_clear: np.ndarray


def analyze_1d(signal: np.ndarray, *, denoise_cutoff: float = 0.10,
               denoise_low_cutoff: float = 0.01,
               denoise_high_cutoff: float = 0.20,
               harmonics_k: int = 16,
               noise_gate: float = 0.0) -> SignalTriplet:
    """Compute the three signals (original / denoised / clear) and their
    magnitude spectra, all on the same length so they can be plotted in
    separate axes without overlap.
    
    For audio denoising, use denoise_low_cutoff and denoise_high_cutoff
    to create a band-pass filter (e.g., isolate commenter voice 500-4400 Hz).
    
    For noise gating, use noise_gate (0 = off, 1 = max suppression).
    Higher values suppress more energy below the noise floor.
    """
    x = signal.astype(np.float32)
    x = x - x.mean()
    n = x.size
    t = np.arange(n)

    F = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    mag = np.abs(F)

    # Denoise: band-pass filter (keep frequencies between low and high cutoffs)
    # This isolates commenter voice and removes crowd noise
    low_bin = max(1, int(denoise_low_cutoff * len(F)))
    high_bin = max(1, int(denoise_high_cutoff * len(F)))
    
    F_dn = F.copy()
    F_dn[:low_bin] = 0           # remove very low frequencies (crowd rumble)
    F_dn[high_bin:] = 0          # remove high frequencies
    
    # Apply spectral gating if enabled (0 = off, 1 = max suppression)
    if noise_gate > 0:
        # Estimate noise floor from denoised signal
        noise_floor = estimate_noise_floor(np.fft.irfft(F_dn, n=n))
        # Scale gate threshold: 0→0 dB (no gating), 1→12 dB (strong gating)
        gate_db = noise_gate * 12.0
        gate_threshold = 10 ** (gate_db / 20.0)
        mag_dn = np.abs(F_dn)
        phase_dn = np.angle(F_dn)
        
        # Suppress bins below noise_floor * gate_threshold
        threshold_mag = noise_floor * gate_threshold
        mask = np.maximum(mag_dn / (threshold_mag + 1e-10), 0.0)
        mask = np.minimum(mask, 1.0)
        F_dn = mask * mag_dn * np.exp(1j * phase_dn)
    
    denoised = np.fft.irfft(F_dn, n=n)

    # "Clear": keep top-K largest magnitude bins (excluding DC)
    K = max(1, min(harmonics_k, len(F) - 1))
    order = np.argsort(mag[1:])[::-1] + 1   # skip DC
    keep = np.zeros_like(F)
    keep[0] = F[0]
    keep[order[:K]] = F[order[:K]]
    clear = np.fft.irfft(keep, n=n)

    return SignalTriplet(
        t=t,
        original=x,
        denoised=denoised.astype(np.float32),
        clear=clear.astype(np.float32),
        freqs=freqs,
        mag_original=mag.astype(np.float32),
        mag_denoised=np.abs(F_dn).astype(np.float32),
        mag_clear=np.abs(keep).astype(np.float32),
    )
