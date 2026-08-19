"""Noise processor: subtle gated room tone injection.

Adds noise only when the music is present. This prevents audible hiss during quiet
intros/outros while still breaking perfectly clean AI noise floors.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from pathlib import Path

class NoiseProcessor:
    """Injects extremely subtle gated room tone / tape hiss."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr

    def process(self, y: np.ndarray, level_db: float = -66.0) -> np.ndarray:
        """Add very low noise floor only when signal is active."""
        amp = 10 ** (level_db / 20.0)
        noise = np.random.normal(0, amp, size=y.shape)

        # Build a smooth gate from the program level so silent/quiet intros stay clean.
        mono = np.mean(np.abs(y), axis=1) if y.ndim > 1 else np.abs(y)
        window = max(1, int(self.sr * 0.05))  # 50ms envelope
        kernel = np.ones(window, dtype=np.float32) / window
        envelope = np.convolve(mono, kernel, mode="same")

        threshold = 10 ** (-45 / 20.0)
        gate = np.clip((envelope - threshold) / (threshold * 4), 0.0, 1.0)

        # Slow fade-in for the very beginning to remove the first-seconds hiss.
        fade_len = min(len(gate), int(self.sr * 2.0))
        if fade_len > 0:
            gate[:fade_len] *= np.linspace(0.0, 1.0, fade_len)

        if y.ndim > 1:
            gate = gate[:, np.newaxis]

        return y + (noise * gate)

    def process_file(self, path: str | Path, level_db: float = -66.0) -> str:
        path_obj = Path(path)
        y, file_sr = sf.read(str(path_obj))
        
        y_out = self.process(y, level_db)

        out_path = path_obj.with_name(f"{path_obj.stem}_noise{path_obj.suffix}")
        sf.write(str(out_path), y_out, file_sr)
        return str(out_path)
