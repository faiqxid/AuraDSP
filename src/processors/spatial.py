"""Spatial processor: convolution reverb + crossfeed."""
from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from pedalboard import Pedalboard, Reverb

class SpatialProcessor:
    """Adds subtle room ambience via built-in reverb."""

    def __init__(self, sr: int = 44100):
        self.sr = sr

    def process(self, y: np.ndarray, mix: float = 0.1, room_size: float = 0.6) -> np.ndarray:
        """Apply subtle reverb to the mix."""
        board = Pedalboard([
            Reverb(room_size=room_size, damping=0.5, wet_level=mix, dry_level=1.0-mix)
        ])
        
        if y.ndim > 1:
            y_in = y.T
        else:
            y_in = y

        y_out = board(y_in, self.sr, reset=False)

        if y_out.ndim > 1:
            y_out = y_out.T

        return y_out

    def process_file(self, path: str | Path, mix: float = 0.1) -> str:
        path_obj = Path(path)
        y, file_sr = sf.read(str(path_obj))

        y_out = self.process(y, mix)

        out_path = path_obj.with_name(f"{path_obj.stem}_spatial{path_obj.suffix}")
        sf.write(str(out_path), y_out, file_sr)
        return str(out_path)
