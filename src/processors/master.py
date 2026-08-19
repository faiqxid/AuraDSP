"""Mastering chain: true LUFS normalize + multiband compressor + final limiter.
Also contains ReferenceMaster for Moises-style reference track matching.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass

import pyloudnorm as pyln
import librosa
import soundfile as sf
from pedalboard import Pedalboard, Compressor, Limiter, LowpassFilter, HighpassFilter, PeakFilter


@dataclass
class ReferenceProfile:
    """Extracted mastering profile from a reference track."""
    lufs: float                          # Integrated LUFS
    peak: float                          # True peak (linear)
    rms: float                           # Overall RMS
    dynamic_range_db: float              # Peak to RMS ratio in dB
    spectral_centroid: float             # Hz
    rms_low: float                       # RMS <200Hz band
    rms_mid: float                       # RMS 200-2000Hz band
    rms_high: float                      # RMS >2000Hz band
    sr: int                              # Sample rate of reference


class ReferenceMaster:
    """Moises-style reference track mastering.

    Extracts loudness, dynamics, and spectral profile from a reference WAV/MP3
    then applies matched mastering to the target audio.
    """

    def __init__(self, sr: int = 44100):
        self.sr = sr

    # ------------------------------------------------------------------
    # Profile extraction
    # ------------------------------------------------------------------

    def _band_rms(self, y: np.ndarray, sr: int, low_cut: float, high_cut: float | None) -> float:
        """Compute RMS of a frequency band."""
        if y.ndim > 1:
            y_mono = y.mean(axis=1)
        else:
            y_mono = y.astype(np.float32)
        
        y_t = y_mono.astype(np.float32)
        
        if low_cut > 0:
            y_t = HighpassFilter(cutoff_frequency_hz=float(low_cut))(y_t, sr)
        if high_cut is not None:
            y_t = LowpassFilter(cutoff_frequency_hz=float(high_cut))(y_t, sr)
        
        rms = float(np.sqrt(np.mean(y_t ** 2))) if y_t.size else 1e-9
        return max(rms, 1e-9)

    def extract_profile(self, reference_path: str | Path) -> dict:
        """Load reference file and extract mastering profile.

        Returns a dict with keys: lufs, peak, rms, spectral_centroid,
        rms_low, rms_mid, rms_high, sr.
        """
        ref_path = str(reference_path)
        
        # Load reference (soundfile for WAV/FLAC, librosa for MP3)
        try:
            y_ref, ref_sr = sf.read(ref_path)
        except Exception:
            y_ref, ref_sr = librosa.load(ref_path, sr=self.sr, mono=False)
            if y_ref.ndim > 1:
                y_ref = y_ref.T  # librosa returns (channels, samples) → (samples, channels)
        
        # Resample to our pipeline SR if needed
        if ref_sr != self.sr:
            if y_ref.ndim > 1:
                y_ref = librosa.resample(y_ref.T, orig_sr=ref_sr, target_sr=self.sr).T
            else:
                y_ref = librosa.resample(y_ref, orig_sr=ref_sr, target_sr=self.sr)
        
        # LUFS
        meter = pyln.Meter(rate=self.sr)
        lufs = meter.integrated_loudness(y_ref)
        if not np.isfinite(lufs):
            lufs = -14.0
        
        # Peak + RMS + Dynamics
        peak = float(np.max(np.abs(y_ref)))
        rms = float(np.sqrt(np.mean(y_ref ** 2)))
        dr_db = 20 * np.log10(max(peak, 1e-9) / max(rms, 1e-9))
        
        # Band RMS
        rms_low = self._band_rms(y_ref, self.sr, 0, 200)
        rms_mid = self._band_rms(y_ref, self.sr, 200, 2000)
        rms_high = self._band_rms(y_ref, self.sr, 2000, None)

        # Spectral centroid (Hz) from mono
        y_mono = y_ref.mean(axis=1) if y_ref.ndim > 1 else y_ref
        centroid = float(np.mean(
            librosa.feature.spectral_centroid(y=y_mono.astype(np.float32), sr=self.sr)
        ))

        return {
            "lufs": lufs,
            "peak": peak,
            "rms": rms,
            "dynamic_range_db": dr_db,
            "spectral_centroid": centroid,
            "rms_low": rms_low,
            "rms_mid": rms_mid,
            "rms_high": rms_high,
            "sr": self.sr,
        }

    # ------------------------------------------------------------------
    # Matching logic
    # ------------------------------------------------------------------

    def _match_band_gain(self, y: np.ndarray, profile: dict) -> np.ndarray:
        """Adjust per-band gain to match reference band RMS.

        Each band's gain is computed from reference_band_rms / input_band_rms,
        then clamped to ±12 dB so we don't explode quiet bands.
        """
        is_stereo = y.ndim > 1
        y_t = y.T if is_stereo else y

        # Extract target band RMS
        target_rms_low = self._band_rms(y, self.sr, 0, 200)
        target_rms_mid = self._band_rms(y, self.sr, 200, 2000)
        target_rms_high = self._band_rms(y, self.sr, 2000, None)

        ref_low = profile["rms_low"]
        ref_mid = profile["rms_mid"]
        ref_high = profile["rms_high"]

        # Gain ratios (clamped to ±12 dB), skip matching if band is effectively silent
        MIN_ACTIVE_RMS = 1e-4  # -80dBFS — below this, band matching is meaningless

        def safe_gain(ref_rms: float, tgt_rms: float) -> float:
            if tgt_rms < MIN_ACTIVE_RMS:
                return 1.0  # don't amplify silent bands
            ratio = ref_rms / tgt_rms
            # Clamp to ±12 dB
            return float(np.clip(ratio, 10 ** (-12/20), 10 ** (12/20)))

        gain_low = safe_gain(ref_low, target_rms_low)
        gain_mid = safe_gain(ref_mid, target_rms_mid)
        gain_high = safe_gain(ref_high, target_rms_high)

        # Build each band, scale, then recombine
        low = LowpassFilter(cutoff_frequency_hz=200)(y_t, self.sr) * gain_low
        mid_raw = HighpassFilter(cutoff_frequency_hz=200)(y_t, self.sr)
        mid = LowpassFilter(cutoff_frequency_hz=2000)(mid_raw, self.sr) * gain_mid
        high = HighpassFilter(cutoff_frequency_hz=2000)(mid_raw, self.sr) * gain_high

        # Sum bands (no extra gain staging — bands don't fully add up due to filter roll-off)
        summed = low + mid + high

        if is_stereo:
            summed = summed.T

        # Safety clip
        peak = float(np.max(np.abs(summed)))
        if peak > 0.95:
            summed = summed / peak * 0.95

        return summed

    def _match_lufs(self, y: np.ndarray, target_lufs: float) -> np.ndarray:
        """Normalize audio to match the target LUFS."""
        meter = pyln.Meter(rate=self.sr)
        loudness = meter.integrated_loudness(y)
        if not np.isfinite(loudness):
            return y
        gain_db = target_lufs - loudness
        gain_linear = 10.0 ** (gain_db / 20.0)
        return y * gain_linear

    def _match_dynamics(self, y: np.ndarray, target_dr_db: float) -> np.ndarray:
        """Apply compression to reduce dynamic range toward a target DR.

        Uses a single-band compressor. Threshold derived from target_dr_db — if
        target is 6dB, threshold is set so peaks above that ratio get compressed.
        Only compresses if current DR significantly exceeds target DR (>3dB gap).

        Args:
            y: Audio (samples, channels) or mono.
            target_dr_db: Target dynamic range in dB from reference profile.

        Returns:
            Compressed audio.
        """
        is_stereo = y.ndim > 1
        y_t = y.T if is_stereo else y

        # Measure current DR
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms < 1e-9:
            return y
        current_dr = 20 * np.log10(peak / rms)

        dr_excess = current_dr - target_dr_db
        if dr_excess < 3.0:
            # Already within range — skip compression
            return y

        # Static soft limiting avoids attack transients that can increase crest factor.
        # Set ceiling from current RMS and requested peak-to-RMS ratio.
        target_peak = rms * 10.0 ** (target_dr_db / 20.0)
        target_peak = float(np.clip(target_peak, 1e-4, 0.95))
        y_limited = target_peak * np.tanh(y_t / target_peak)

        # Blend based on DR excess; prevents over-flattening natural material.
        blend = float(np.clip(dr_excess / 12.0, 0.25, 1.0))
        y_compressed = y_t * (1.0 - blend) + y_limited * blend

        if is_stereo:
            y_compressed = y_compressed.T

        # Safety clip
        peak_c = float(np.max(np.abs(y_compressed)))
        if peak_c > 0.95:
            y_compressed = y_compressed / peak_c * 0.95

        return y_compressed

    def process(self, y: np.ndarray, reference_path: str | Path) -> np.ndarray:
        """Apply reference-matched mastering to target audio.

        pipeline:
          1. Extract reference profile
          2. Band-gain matching (EQ shape)
          3. Dynamic range matching (compression)
          4. LUFS normalize to reference LUFS
          5. Safety peak clip at 0.95

        Args:
            y: Target audio (samples, channels) or (samples,) mono.
            reference_path: Path to reference WAV/MP3 file.

        Returns:
            Processed audio same shape as input.
        """
        profile = self.extract_profile(reference_path)

        # Step 1: Band gain matching
        y_out = self._match_band_gain(y, profile)

        # Step 2: Dynamic range matching
        y_out = self._match_dynamics(y_out, profile["dynamic_range_db"])

        # Step 3: LUFS normalize to reference loudness
        y_out = self._match_lufs(y_out, profile["lufs"])

        # Step 4: Safety clip
        peak = float(np.max(np.abs(y_out)))
        if peak > 0.95:
            y_out = y_out / peak * 0.95

        return y_out

    def process_file(self, input_path: str | Path, reference_path: str | Path,
                     output_path: str | Path | None = None) -> str:
        """Apply reference mastering to a file and save result.

        Args:
            input_path: Input audio file.
            reference_path: Reference audio file.
            output_path: Optional output path. Defaults to input_stem_ref.wav.

        Returns:
            Path to the output file.
        """
        in_path = Path(input_path)
        y, sr = sf.read(str(in_path))

        if sr != self.sr:
            if y.ndim > 1:
                y = librosa.resample(y.T, orig_sr=sr, target_sr=self.sr).T
            else:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)

        y_out = self.process(y, reference_path)

        out = Path(output_path) if output_path else in_path.with_name(f"{in_path.stem}_ref.wav")
        sf.write(str(out), y_out, self.sr)
        return str(out)


class MasterChain:
    """Final mastering: LUFS normalization → multiband comp → final limiter."""

    def __init__(self, sr: int = 44100, target_lufs: float = -14.0):
        self.sr = sr
        self.target_lufs = target_lufs

    def normalize_lufs(self, y: np.ndarray, target_lufs: float | None = None) -> np.ndarray:
        """True LUFS normalization via pyloudnorm (ITU-R BS.1770-4).

        Args:
            y: Audio array (samples, channels) or (samples,) mono.
            target_lufs: Target LUFS. Defaults to self.target_lufs (-14 LUFS).

        Returns:
            Gain-adjusted audio matching target LUFS.
        """
        target = target_lufs if target_lufs is not None else self.target_lufs
        meter = pyln.Meter(rate=self.sr)
        # pyloudnorm expects (samples, channels) — same as soundfile convention
        y_meter = y
        loudness = meter.integrated_loudness(y_meter)
        if not np.isfinite(loudness):
            return y
        gain_db = target - loudness
        gain_linear = 10.0 ** (gain_db / 20.0)
        return y * gain_linear

    @staticmethod
    def _safe_peak(y: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > ceiling:
            return y / peak * ceiling
        return y

    def _multiband(self, y: np.ndarray) -> np.ndarray:
        """Three-band compressor: Low (<200Hz), Mid (200-2000Hz), High (>2000Hz).

        Each band gets its own compression settings. High band is more aggressive
        to destroy AI checkerboard artifacts.
        """
        if y.ndim > 1:
            y = y.T

        # Split into 3 bands
        low = LowpassFilter(cutoff_frequency_hz=200)(y, self.sr)
        mid_raw = HighpassFilter(cutoff_frequency_hz=200)(y, self.sr)
        mid = LowpassFilter(cutoff_frequency_hz=2000)(mid_raw, self.sr)
        high = HighpassFilter(cutoff_frequency_hz=2000)(mid_raw, self.sr)

        # Band-independent compression
        comp_low = Compressor(threshold_db=-18.0, ratio=3.0, attack_ms=5.0, release_ms=50.0)
        comp_mid = Compressor(threshold_db=-22.0, ratio=3.0, attack_ms=0.1, release_ms=100.0)
        comp_high = Compressor(threshold_db=-20.0, ratio=3.0, attack_ms=0.1, release_ms=80.0)

        low_c = comp_low(low, self.sr, reset=False)
        mid_c = comp_mid(mid, self.sr, reset=False)
        high_c = comp_high(high, self.sr, reset=False)

        # Recombine with gain staging
        gain = 1.0 / np.sqrt(3)
        summed = low_c * gain + mid_c * gain + high_c * gain

        # Clip-safe
        summed = self._safe_peak(summed, ceiling=0.95)

        if y.ndim > 1:
            summed = summed.T
        return summed

    def _final_master(self, y: np.ndarray) -> np.ndarray:
        """Final single-band compressor + true-peak limiter."""
        if y.ndim > 1:
            y = y.T

        board = Pedalboard([
            Compressor(threshold_db=-24.0, ratio=4.0, attack_ms=0.1, release_ms=100.0),
            Limiter(threshold_db=-1.0, release_ms=100.0),
        ])
        y_out = board(y, self.sr, reset=False)

        if y.ndim > 1:
            y_out = y_out.T

        return self._safe_peak(y_out, ceiling=0.95)

    def process(self, y: np.ndarray) -> np.ndarray:
        """Full mastering chain.

        Steps:
          1. Multiband compressor (3 bands)
          2. Final compressor + limiter
          3. True LUFS normalization (-14 LUFS) — LAST to ensure target hit
        """
        # Step 1: Multiband compression
        y = self._multiband(y)

        # Step 2: Final master
        y = self._final_master(y)

        # Step 3: LUFS normalize (after all dynamic processing)
        y = self.normalize_lufs(y)

        # Final safety clip
        y = self._safe_peak(y, ceiling=0.95)

        return y

    def process_file(self, path: str | Path) -> str:
        import soundfile as sf

        path_obj = Path(path)
        y, file_sr = sf.read(str(path_obj))

        y_out = self.process(y)

        out_path = path_obj.with_name(f"{path_obj.stem}_master{path_obj.suffix}")
        sf.write(str(out_path), y_out, file_sr)
        return str(out_path)