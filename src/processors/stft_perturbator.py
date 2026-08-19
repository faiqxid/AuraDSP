"""Spectral processor: STFT spectral smoothing to destroy AI checkerboard artifacts.

The lofcz detector looks for periodic spectral peaks at 1-8kHz from deconvolution.
Spectral smoothing (convolution across frequency axis) smears these peaks into
adjacent bins, destroying the periodicity while sounding like a subtle warm EQ.
"""
from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

class SpectralPerturbator:
    """STFT spectral smoothing — directly attacks checkerboard periodicity naturally."""
    
    def __init__(self, sr: int = 44100, n_fft: int = 8192):
        """Use n_fft=8192 to match lofcz detector's analysis resolution."""
        self.sr = sr
        self.n_fft = n_fft
        
    def process(self, y: np.ndarray, amount: float = 0.15) -> np.ndarray:
        """Apply spectral smoothing to 1-8kHz range.
        
        amount: smoothing strength (0.05=subtle, 0.15=moderate, 0.25=strong).
        Higher values smear peaks more aggressively.
        """
        if y.ndim > 1:
            return np.array([self.process(ch, amount) for ch in y])
            
        hop = self.n_fft // 4
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=hop)
        mag, phase = np.abs(stft), np.angle(stft)
        
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        target_bins = (freqs >= 1000) & (freqs <= 8000)
        
        if not np.any(target_bins):
            return y
            
        # Calculate smoothing kernel size based on amount
        # At 8192 FFT on 44100Hz: bin spacing ~5.4Hz
        # A kernel of 20 bins ≈ 108Hz smoothing — enough to smear checkerboard peaks
        kernel_size = max(3, int(amount * 200))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = np.ones(kernel_size) / kernel_size
        
        # Smooth ONLY the target frequency range across frequency axis
        mag_smoothed = mag.copy()
        target_indices = np.where(target_bins)[0]
        
        for i in range(mag.shape[1]):  # For each frame
            frame = mag[target_bins, i]
            smoothed = np.convolve(frame, kernel, mode='same')
            # Blend: original * (1-amount) + smoothed * amount
            mag_smoothed[target_bins, i] = frame * (1 - amount * 3) + smoothed * (amount * 3)
        
        # Ensure no negative magnitudes
        mag_smoothed = np.maximum(mag_smoothed, 0)
        
        # Reconstruct
        stft_new = mag_smoothed * (np.cos(phase) + 1j * np.sin(phase))
        y_out = librosa.istft(stft_new, hop_length=hop, length=len(y))
        
        # Normalize to original level
        peak_orig = np.max(np.abs(y)) + 1e-10
        peak_new = np.max(np.abs(y_out)) + 1e-10
        if peak_new > 0.01:
            y_out = y_out / peak_new * peak_orig
        
        return y_out
        
    def process_file(self, path: str | Path, amount: float = 0.15) -> str:
        path_obj = Path(path)
        y, file_sr = librosa.load(str(path_obj), sr=self.sr, mono=False)
        y_out = self.process(y, amount)
        
        if y_out.ndim > 1:
            y_out = y_out.T
            
        out_path = path_obj.with_name(f"{path_obj.stem}_stft_perturb{path_obj.suffix}")
        sf.write(str(out_path), y_out, file_sr)
        return str(out_path)