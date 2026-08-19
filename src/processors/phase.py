"""Phase perturbation via STFT to disrupt phase coherence.

STFT-based approach: each time-frequency bin gets its own smoothed random phase
offset. This avoids the intermodulation distortion that occurs when modulating
the single instantaneous phase of a broadband Hilbert analytic signal.
"""
from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

class PhasePerturbator:
    """Injects smoothed phase noise per frequency bin via STFT."""

    def __init__(self, sr: int = 44100, n_fft: int = 2048):
        self.sr = sr
        self.n_fft = n_fft

    def process(self, y: np.ndarray, amount: float = 0.05) -> np.ndarray:
        """Apply per-bin phase perturbation via STFT.

        amount: max phase offset in radians (0.05 ≈ 2.9 degrees — subtle).
        """
        if y.ndim > 1:
            return np.array([self.process(ch, amount) for ch in y])

        # Short-Time Fourier Transform
        hop = self.n_fft // 4
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=hop)
        mag, phase = np.abs(stft), np.angle(stft)

        # Generate smoothed random phase offsets per frequency bin
        # Shape: (n_freq, n_frames)
        n_freq, n_frames = phase.shape
        raw = np.random.uniform(-np.pi, np.pi, size=phase.shape)

        # Smooth across TIME (axis=1) — prevents crackle
        time_window = max(1, int(n_frames * 0.02))  # ~2% of total frames
        kernel = np.ones(time_window) / time_window
        for f in range(n_freq):
            raw[f] = np.convolve(raw[f], kernel, mode='same')

        # Smooth across FREQUENCY (axis=0) — prevents harsh buzzy artifacts
        freq_window = max(1, int(n_freq * 0.05))
        fkernel = np.ones(freq_window) / freq_window
        for t in range(n_frames):
            raw[:, t] = np.convolve(raw[:, t], fkernel, mode='same')

        # Scale noise to desired amount and apply
        noise = raw * amount
        phase_perturbed = phase + noise

        # Reconstruct
        stft_new = mag * (np.cos(phase_perturbed) + 1j * np.sin(phase_perturbed))
        y_out = librosa.istft(stft_new, hop_length=hop, length=len(y))

        # Safety: normalize to original peak
        peak_orig = np.max(np.abs(y)) + 1e-10
        peak_new = np.max(np.abs(y_out)) + 1e-10
        y_out = y_out / peak_new * peak_orig

        return y_out

    def process_file(self, path: str | Path, amount: float = 0.05) -> str:
        path_obj = Path(path)
        y, file_sr = librosa.load(str(path_obj), sr=self.sr, mono=False)
        y_out = self.process(y, amount)

        if y_out.ndim > 1:
            y_out = y_out.T

        out_path = path_obj.with_name(f"{path_obj.stem}_phase{path_obj.suffix}")
        sf.write(str(out_path), y_out, file_sr)
        return str(out_path)
