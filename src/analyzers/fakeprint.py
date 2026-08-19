"""Fakeprint Extractor — Exact implementation from lofcz/ai-music-detector.

Reference:
  https://github.com/lofcz/ai-music-detector/blob/master/src/python/inference.py
  https://huggingface.co/lofcz/ai-music-detector

Neural vocoders use transposed convolution (deconvolution) layers that create
periodic spectral peaks. This extractor isolates those peaks by subtracting a
lower envelope (hull) from the average power spectrum.
"""

import json
import warnings
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass
from scipy.ndimage import minimum_filter1d

# Suppress librosa decoding warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")


@dataclass
class PreprocessingConfig:
    sample_rate: int = 16000
    n_fft: int = 8192
    freq_min: int = 1000
    freq_max: int = 8000
    hull_area: int = 10
    max_db: float = 5.0
    min_db: float = -45.0
    max_duration: int = 300

    @classmethod
    def from_json(cls, path: str) -> "PreprocessingConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            sample_rate=data.get("sample_rate", 16000),
            n_fft=data.get("n_fft", 8192),
            freq_min=data.get("freq_min", 1000),
            freq_max=data.get("freq_max", 8000),
            hull_area=data.get("hull_area", 10),
            max_db=data.get("max_db", 5.0),
            min_db=data.get("min_db", -45.0),
            max_duration=data.get("max_duration", 300),
        )


def extract_fakeprint(
    audio_path: str,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    """Extract spectral fakeprint from an audio file (matches lofcz reference).

    Args:
        audio_path: Path to audio file (mp3, wav, flac, etc.)
        config: Preprocessing parameters (defaults to lofcz standard)

    Returns:
        numpy array of shape (3585,) — the fakeprint feature vector
    """
    if config is None:
        config = PreprocessingConfig()

    # 1. Load audio, resample to 16kHz, mono
    y, sr = librosa.load(
        audio_path,
        sr=config.sample_rate,
        mono=True,
        duration=config.max_duration,
    )

    if len(y) < config.n_fft:
        y = np.pad(y, (0, config.n_fft - len(y) + 1))

    # 2. STFT power spectrum (not normalized, power=2)
    # Reference: torchaudio.transforms.Spectrogram(n_fft=8192, power=2, normalized=False)
    # hop_length default = n_fft // 4 = 2048
    hop_length = config.n_fft // 4
    spec = np.abs(librosa.stft(y, n_fft=config.n_fft, hop_length=hop_length)) ** 2

    # 3. Convert to dB: 10 * log10(spec)
    # Reference clamps to [1e-10, 1e6] before log
    spec_clamped = np.clip(spec, 1e-10, 1e6)
    spec_db = 10.0 * np.log10(spec_clamped)

    # 4. Average across time
    mean_spectrum = np.mean(spec_db, axis=1)

    # 5. Frequency bins
    freqs = librosa.fft_frequencies(sr=config.sample_rate, n_fft=config.n_fft)

    # 6. Crop to 1-8 kHz band
    freq_mask = (freqs >= config.freq_min) & (freqs <= config.freq_max)
    freq_spectrum = mean_spectrum[freq_mask]

    # 7. Lower hull using minimum filter
    hull = minimum_filter1d(freq_spectrum, size=config.hull_area, mode="nearest")

    # 8. Clip hull to min_db
    hull = np.clip(hull, config.min_db, None)

    # 9. Residue: spectrum - hull, clip negative values
    residue = np.clip(freq_spectrum - hull, 0, None)

    # 10. Clip residue to max_db and normalize by max
    residue = np.clip(residue, 0, config.max_db)
    max_val = np.max(residue) + 1e-6
    fakeprint = residue / max_val

    # 11. Ensure exact feature count (3585)
    target_len = 3585
    if len(fakeprint) < target_len:
        fakeprint = np.pad(fakeprint, (0, target_len - len(fakeprint)))
    elif len(fakeprint) > target_len:
        fakeprint = fakeprint[:target_len]

    return fakeprint.astype(np.float32)
