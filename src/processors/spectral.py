"""Spectral processor: saturation, excitation, and non-linear harmonic distortion.

Disrupts the checkerboard spectral pattern leftover from deconvolution layers.
Uses per-stem FFmpeg vibrato on non-vocal stems only in aggressive mode.
"""
from __future__ import annotations

import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
from pedalboard import Pedalboard, Distortion, Compressor, HighpassFilter, LowpassFilter, Chorus, Phaser

class SpectralProcessor:
    """Applies pedalboard DSP to stems to disrupt AI spectral fingerprints."""
    
    def __init__(self, sr: int = 44100, intensity: str = "quality"):
        self.sr = sr
        self.intensity = intensity
        self.drive_scale = 1.0 if intensity == "quality" else 1.9
        
    def _limit_drive(self, drive_db: float) -> float:
        """Scale distortion safely for aggressive mode."""
        return min(drive_db * self.drive_scale, 5.0)
        
    def _get_vocal_chain(self) -> Pedalboard:
        chain = [
            HighpassFilter(cutoff_frequency_hz=80.0),
            Compressor(threshold_db=-15.0, ratio=3.0, release_ms=100.0),
            Distortion(drive_db=self._limit_drive(1.5)),
        ]
        if self.intensity == "aggressive":
            # NO vibrato on vocals — keeps natural human sound
            chain.append(Chorus(rate_hz=0.8, depth=0.3, centre_delay_ms=8.0, feedback=0.05, mix=0.2))
            chain.append(Phaser(rate_hz=0.5, depth=0.3, feedback=0.05, mix=0.15))
        return Pedalboard(chain)
        
    def _get_drum_chain(self) -> Pedalboard:
        chain = [
            Compressor(threshold_db=-20.0, ratio=4.0, attack_ms=5.0, release_ms=50.0),
            Distortion(drive_db=self._limit_drive(2.0))
        ]
        return Pedalboard(chain)
        
    def _get_bass_chain(self) -> Pedalboard:
        chain = [
            LowpassFilter(cutoff_frequency_hz=800.0),
            Distortion(drive_db=self._limit_drive(2.5)),
            Compressor(threshold_db=-12.0, ratio=4.0)
        ]
        if self.intensity == "aggressive":
            chain.append(Chorus(rate_hz=0.4, depth=0.2, centre_delay_ms=5.0, feedback=0.05, mix=0.15))
        return Pedalboard(chain)
        
    def _get_other_chain(self) -> Pedalboard:
        chain = [
            Chorus(rate_hz=0.5, depth=0.5, centre_delay_ms=7.0, feedback=0.1, mix=0.4),
            Distortion(drive_db=self._limit_drive(1.0))
        ]
        if self.intensity == "aggressive":
            chain.append(Phaser(rate_hz=0.7, depth=0.5, feedback=0.05, mix=0.3))
        return Pedalboard(chain)
        
    def _apply_vibrato(self, input_path: str, output_path: str, rate: float = 2.0, depth: float = 0.05):
        """Apply FFmpeg vibrato to a non-vocal stem only."""
        import os
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", f"vibrato=f={rate}:d={depth}",
            str(output_path)
        ]
        kwargs = {"capture_output": True, "check": True}
        if os.name == "nt":
            import subprocess
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.run(cmd, **kwargs)

    def process_file(self, path: str | Path, stem_type: str) -> str:
        """Apply spectral processing to a stem file."""
        path_obj = Path(path)
        y, sr = sf.read(str(path_obj))
        
        if stem_type == "vocals":
            board = self._get_vocal_chain()
        elif stem_type == "drums":
            board = self._get_drum_chain()
        elif stem_type == "bass":
            board = self._get_bass_chain()
        else:
            board = self._get_other_chain()
            
        # pedalboard expects (channels, samples). soundfile gives (samples, channels)
        if y.ndim > 1:
            y_in = y.T
        else:
            y_in = y
            
        y_out = board(y_in, self.sr, reset=False)
        
        # Back to soundfile shape
        if y_out.ndim > 1:
            y_out = y_out.T
            
        # Safety peak limit after aggressive DSP
        peak = float(np.max(np.abs(y_out))) if y_out.size else 0.0
        if peak > 0.85:
            y_out = y_out / peak * 0.85
            
        out_path = path_obj.with_name(f"{path_obj.stem}_spectral{path_obj.suffix}")
        sf.write(str(out_path), y_out, sr)
        
        # Apply vibrato to non-vocal stems + natural-rate vibrato on vocals
        # Real singer vibrato: 5-6Hz rate, 0.3-0.5 semitones depth
        if self.intensity == "aggressive":
            vib_path = str(out_path).replace("_spectral", "_vib")
            try:
                if stem_type == "vocals":
                    # Natural vocal vibrato: 5Hz, very subtle depth (sounds like real singer)
                    self._apply_vibrato(str(out_path), vib_path, rate=5.0, depth=0.02)
                elif stem_type in ("other", "bass"):
                    # Guitar/piano vibrato: 2Hz, slightly deeper
                    self._apply_vibrato(str(out_path), vib_path, rate=2.0, depth=0.05)
                else:
                    # Drums: no vibrato (transient content)
                    return str(out_path)
                return vib_path
            except Exception:
                pass
        
        return str(out_path)
