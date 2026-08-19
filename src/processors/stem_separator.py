"""Stem Separation using audio-separator (Demucs).

Reuses ONE shared Separator to prevent CUDA OOM. Uses short temp path for MAX_PATH.
Suppresses audio-separator logging (colorama crash in background threads).
"""
from __future__ import annotations

import gc
import logging
import os
import shutil
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

# Silence audio-separator loggers
for _n in ("separator", "common_separator", "demucs_separator"):
    _l = logging.getLogger(_n)
    _l.disabled = True


@dataclass
class SeparationResult:
    vocals_path: str
    drums_path: str
    bass_path: str
    other_path: str
    extra_paths: list[str] = field(default_factory=list)


# === Global shared Separator — prevents CUDA OOM from session accumulation ===
_SHARED_SEP = None
_SHORT_DIR = Path("C:/tmp/mar")


def _purge_demucs_temp():
    """Purge stale stems from C:\\tmp\\mar\\out\\ on startup to ensure clean slate."""
    out_dir = _SHORT_DIR / "out"
    if out_dir.exists():
        for f in out_dir.glob("*"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass


def _get_shared_separator():
    global _SHARED_SEP
    if _SHARED_SEP is None:
        from audio_separator.separator import Separator
        _SHORT_DIR.mkdir(parents=True, exist_ok=True)
        _purge_demucs_temp()
        (_SHORT_DIR / "out").mkdir(exist_ok=True)
        _SHARED_SEP = Separator(
            output_dir=str(_SHORT_DIR / "out"),
            output_format="WAV",
            log_level=100,
        )
        _SHARED_SEP.load_model(model_filename="htdemucs_6s.yaml")
    return _SHARED_SEP


class StemSeparator:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_filename = "htdemucs_6s.yaml"

    def _merge_stems(self, paths: list[str], output_name: str) -> str:
        total = None
        sr = None
        for p in paths:
            y, sr = sf.read(p, dtype="float32")
            if y.ndim == 1:
                y = y[:, np.newaxis]
            if total is None:
                total = y
                continue
            max_len = max(total.shape[0], y.shape[0])
            n_ch = max(total.shape[1], y.shape[1])
            def pad(a):
                if a.shape[1] < n_ch: a = np.tile(a, (1, n_ch))
                if a.shape[0] < max_len: a = np.pad(a, ((0, max_len - a.shape[0]), (0, 0)))
                return a
            total = pad(total) + pad(y)
        if total is None: raise RuntimeError("No stems to merge")
        peak = float(np.max(np.abs(total))) if total.size else 0.0
        if peak > 0.95: total = total / peak * 0.95
        out = self.output_dir / output_name
        sf.write(str(out), total, sr)
        return str(out)

    def separate(self, audio_path: str | Path) -> SeparationResult:
        audio_path = Path(audio_path)
        inp = audio_path

        # If path is long (>120 chars), copy to short temp to avoid MAX_PATH
        if len(str(audio_path)) > 120:
            _SHORT_DIR.mkdir(parents=True, exist_ok=True)
            sn = audio_path.stem[:20].replace(" ", "_") + audio_path.suffix
            inp = _SHORT_DIR / sn
            shutil.copy2(str(audio_path), str(inp))

        sep = _get_shared_separator()
        filenames = sep.separate(str(inp))

        # Clean input copy
        if inp != audio_path:
            inp.unlink(missing_ok=True)

        if not filenames:
            raise RuntimeError(f"Separator returned no stems for {audio_path.name}")

        stems: dict[str, str] = {}
        other_bucket: list[str] = []
        output_files: list[str] = []

        for fp in filenames:
            src = Path(fp)
            # audio-separator returns relative filenames from output_dir
            if not src.is_absolute():
                src = _SHORT_DIR / "out" / fp
            if not src.exists():
                raise RuntimeError(f"Stem file not found: {src}")

            dst = self.output_dir / src.name
            shutil.copy2(str(src), str(dst))
            output_files.append(str(dst))
            n = dst.name.lower()
            if "vocals" in n: stems["vocals"] = str(dst)
            elif "drums" in n: stems["drums"] = str(dst)
            elif "bass" in n: stems["bass"] = str(dst)
            elif any(k in n for k in ("other", "guitar", "piano")):
                other_bucket.append(str(dst))

        # Clean shared output dir for next run
        for f in (_SHORT_DIR / "out").glob("*.wav"):
            f.unlink(missing_ok=True)

        if other_bucket:
            stems["other"] = self._merge_stems(other_bucket, "merged_other_htdemucs_6s.wav")

        missing = [k for k in ("vocals", "drums", "bass", "other") if k not in stems]
        if missing:
            raise RuntimeError(
                f"Missing stems: {missing}. Files: {output_files}"
            )

        return SeparationResult(
            vocals_path=stems["vocals"],
            drums_path=stems["drums"],
            bass_path=stems["bass"],
            other_path=stems["other"],
            extra_paths=other_bucket,
        )
