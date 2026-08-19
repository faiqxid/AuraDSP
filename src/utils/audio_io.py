"""Audio I/O utility — load audio files for analysis. MP3-safe, quiet."""
import os
import sys
import contextlib
import warnings
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass

# Suppress Python-level warnings
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*Giving up searching.*")
warnings.filterwarnings("ignore", message=".*get_duration.*keyword argument.*")
warnings.filterwarnings("ignore", message=".*librosa.core.audio.*")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


@contextlib.contextmanager
def _suppress_stderr():
    """Suppress C-level stderr (mpg123, ffmpeg spam) during audio loading."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)


@dataclass
class AudioInfo:
    filepath: str
    sample_rate: int
    duration: float
    channels: int
    format: str
    bit_depth: int | None = None


def load_audio(filepath: str, sr: int = 22050, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio file, return (samples, sample_rate).
    WAV/FLAC/OGG -> soundfile (fast).
    MP3/M4A/etc -> librosa (ffmpeg fallback).
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = p.suffix.lower()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if ext in {".wav", ".flac", ".ogg"}:
            data, file_sr = sf.read(str(filepath), dtype="float32")
            if data.ndim == 1:
                data = data[:, np.newaxis]
            data = data.T
        else:
            import librosa as _lb
            with _suppress_stderr():
                data, file_sr = _lb.load(str(filepath), sr=None, mono=False)

    # Ensure (channels, samples)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    # Mix to mono if needed
    if mono and data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    elif mono:
        data = data[0:1, :]

    # Resample if needed
    if file_sr != sr:
        import librosa as _lb
        with _suppress_stderr():
            if mono:
                data = _lb.resample(data[0], orig_sr=file_sr, target_sr=sr)[np.newaxis, :]
            else:
                resampled = []
                for ch in range(data.shape[0]):
                    resampled.append(_lb.resample(data[ch], orig_sr=file_sr, target_sr=sr))
                data = np.stack(resampled)
        file_sr = sr

    if mono:
        return data[0], file_sr
    return data, file_sr


def get_audio_info(filepath: str) -> AudioInfo:
    """Get metadata about an audio file without triggering spam."""
    p = Path(filepath)
    ext = p.suffix.lower()

    if ext in {".wav", ".flac", ".ogg"}:
        info = sf.info(str(filepath))
        return AudioInfo(
            filepath=str(p.absolute()),
            sample_rate=info.samplerate,
            duration=info.duration,
            channels=info.channels,
            format=info.format,
            bit_depth=info.subtype_info.split()[0] if info.subtype_info else None,
        )
    else:
        import librosa as _lb
        with warnings.catch_warnings(), _suppress_stderr():
            warnings.simplefilter("ignore")
            dur = _lb.get_duration(path=str(filepath))
        return AudioInfo(
            filepath=str(p.absolute()),
            sample_rate=0,
            duration=dur,
            channels=0,
            format=p.suffix.strip(".").upper(),
        )
