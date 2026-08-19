"""Humanizer processor: timing and pitch randomization.

Breaks mathematical precision (timing grid, perfect pitch) to avoid AI pattern matching.
"""
from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

class Humanizer:
    """Adds non-linear micro-pitch drift and onset timing shifts."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        
    def process_vocals(self, y: np.ndarray, max_drift_cents: float = 5.0) -> np.ndarray:
        """Apply random-walk micro-pitch drift to vocals."""
        if y.ndim > 1:
            return np.array([self.process_vocals(ch, max_drift_cents) for ch in y])
            
        shift_semitones = np.random.uniform(-max_drift_cents, max_drift_cents) / 100.0
        y_shifted = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=shift_semitones)
        return y_shifted
        
    def process_drums(self, y: np.ndarray, shift_ms: float = 5.0) -> np.ndarray:
        """Apply timing shifts with proper fades at BOTH boundaries."""
        shift_samples = int(self.sr * (shift_ms / 1000.0))
        shift_samples = shift_samples if np.random.random() > 0.5 else -shift_samples
        fade_len = min(64, abs(shift_samples)) if abs(shift_samples) > 0 else 0

        y_shifted = np.zeros_like(y)
        
        if shift_samples > 0:
            # Shift right: content starts at shift_samples
            if y.ndim > 1:
                for ch in range(y.shape[0]):
                    y_shifted[ch, shift_samples:] = y[ch, :-shift_samples]
                    # Fade IN at start of content
                    if fade_len > 0:
                        fade_in = np.linspace(0, 1, fade_len)
                        y_shifted[ch, shift_samples:shift_samples + fade_len] *= fade_in
                    # Fade OUT at end of content (before zeros)
                    if fade_len > 0:
                        fade_out = np.linspace(1, 0, fade_len)
                        y_shifted[ch, -fade_len:] *= fade_out
            else:
                y_shifted[shift_samples:] = y[:-shift_samples]
                if fade_len > 0:
                    fade_in = np.linspace(0, 1, fade_len)
                    y_shifted[shift_samples:shift_samples + fade_len] *= fade_in
                    fade_out = np.linspace(1, 0, fade_len)
                    y_shifted[-fade_len:] *= fade_out
        elif shift_samples < 0:
            abs_shift = abs(shift_samples)
            # Shift left: content starts at 0
            if y.ndim > 1:
                for ch in range(y.shape[0]):
                    y_shifted[ch, :-abs_shift] = y[ch, abs_shift:]
                    # Fade IN at very start
                    if fade_len > 0:
                        fade_in = np.linspace(0, 1, fade_len)
                        y_shifted[ch, :fade_len] *= fade_in
                    # Fade OUT at end of content
                    if fade_len > 0:
                        fade_out = np.linspace(1, 0, fade_len)
                        y_shifted[ch, -abs_shift-fade_len:-abs_shift] *= fade_out
            else:
                y_shifted[:-abs_shift] = y[abs_shift:]
                if fade_len > 0:
                    fade_in = np.linspace(0, 1, fade_len)
                    y_shifted[:fade_len] *= fade_in
                    fade_out = np.linspace(1, 0, fade_len)
                    y_shifted[-abs_shift-fade_len:-abs_shift] *= fade_out
        else:
            y_shifted = y.copy()
                
        return y_shifted
        
    def process_bass(self, y: np.ndarray, drift_cents: float = 3.0) -> np.ndarray:
        """Slight pitch drift and rounding for bass."""
        return self.process_vocals(y, drift_cents)

    def _soft_clip(self, y: np.ndarray, ceiling: float = 0.85) -> np.ndarray:
        """Soft-clip to prevent distortion-induced clicks from exceeding 1.0."""
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > ceiling:
            return y / peak * ceiling
        return y

    def process_file(self, path: str | Path, stem_type: str, drift_cents: float = 5.0, shift_ms: float = 5.0) -> str:
        """Process an audio file directly based on stem type."""
        path_obj = Path(path)
        y, file_sr = librosa.load(str(path_obj), sr=self.sr, mono=False)
        
        if stem_type == "vocals":
            y_out = self.process_vocals(y, max_drift_cents=drift_cents)
        elif stem_type == "drums":
            y_out = self.process_drums(y, shift_ms=shift_ms)
        elif stem_type == "bass":
            y_out = self.process_bass(y, drift_cents=max(3.0, drift_cents * 0.6))
        else:
            y_out = self.process_drums(y, shift_ms=max(2.0, shift_ms * 0.5))
            
        # Prevent clipping from pitch shift or distortion
        y_out = self._soft_clip(y_out, ceiling=0.85)
            
        out_path = path_obj.with_name(f"{path_obj.stem}_humanized{path_obj.suffix}")
        
        # librosa loads (channels, samples). soundfile expects (samples, channels)
        if y_out.ndim > 1:
            y_out = y_out.T
            
        sf.write(str(out_path), y_out, file_sr)
        return str(out_path)
