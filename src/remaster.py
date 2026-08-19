"""Remaster Orchestrator — ties all 6 pipeline steps together.

Steps:
1. Metadata Purge -> 2. Stem Separation -> 3. Per-Stem Humanization
4. Spectral Disruption -> 5. Phase Noise -> 6. Recombine + Spatial + Master
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict

import librosa
import numpy as np
import soundfile as sf

from utils.audio_io import load_audio

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


class CancellationError(RuntimeError):
    """Raised when a shutdown_event is set mid-pipeline to abort processing cleanly."""


@dataclass
class RemasterResult:
    filepath: str
    original_path: str
    ai_score_before: float | None = None
    ai_score_after: float | None = None
    bypass_success: bool | None = None
    notes: list[str] | None = None


class RemasterPipeline:
    """Orchestrates the full 6-step remaster pipeline."""

    def __init__(self, output_dir: str | Path, sample_rate: int = 44100, use_cuda: bool = True, intensity: str = "aggressive",
                 shutdown_event: threading.Event | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sr = sample_rate
        self.use_cuda = use_cuda
        self.intensity = intensity
        self.shutdown_event = shutdown_event

        # Lazy imports for torch-dependent processors
        self._stem_sep = None
        self._humanizer = None
        self._spectral = None
        self._phase = None
        self._spatial = None
        self._noise = None
        self._master = None

    def _check_cancel(self) -> None:
        """Abort at a safe pipeline boundary when desktop shutdown is requested."""
        if self.shutdown_event and self.shutdown_event.is_set():
            raise CancellationError("Remaster cancelled: application is shutting down")

    def _get_stem_sep(self):
        if self._stem_sep is None:
            from processors import get_stem_separator
            StemSep = get_stem_separator()
            self._stem_sep = StemSep(
                output_dir=str(self.output_dir / "stems")
            )
        return self._stem_sep

    def _get_humanizer(self):
        if self._humanizer is None:
            from processors import Humanizer
            self._humanizer = Humanizer(sr=self.sr)
        return self._humanizer

    def _get_spectral(self):
        if self._spectral is None:
            from processors import SpectralProcessor
            self._spectral = SpectralProcessor(sr=self.sr, intensity=self.intensity)
        return self._spectral

    def _get_phase(self):
        if self._phase is None:
            from processors import PhasePerturbator
            self._phase = PhasePerturbator(sr=self.sr)
        return self._phase

    def _get_spatial(self):
        if self._spatial is None:
            from processors import SpatialProcessor
            self._spatial = SpatialProcessor(sr=self.sr)
        return self._spatial

    def _get_noise(self):
        if self._noise is None:
            from processors import NoiseProcessor
            self._noise = NoiseProcessor(sr=self.sr)
        return self._noise

    def _get_master(self):
        if self._master is None:
            from processors import MasterChain
            self._master = MasterChain(sr=self.sr)
        return self._master

    def _strip_metadata(self, input_path: str | Path) -> str:
        """Step 1: Strip all metadata, re-encode to clean WAV."""
        input_path = Path(input_path)
        clean_path = self.output_dir / f"clean_{input_path.stem}.wav"

        try:
            from mutagen import File
            audio = File(str(input_path))
            if audio is not None:
                audio.delete()
                audio.save()
        except Exception:
            pass

        # Re-encode to clean WAV
        y, sr = librosa.load(str(input_path.absolute()), sr=self.sr, mono=False)
        if y.ndim > 1:
            y = y.T
        sf.write(str(clean_path.absolute()), y, sr)
        return str(clean_path.absolute())

    def _ffmpeg_highfreq_vibrato(self, input_path: str, output_path: str):
        """Apply vibrato only to high frequencies (>2kHz) to break AI checkerboard.
        Low frequencies (vocal body, bass, kick) remain untouched for natural sound."""
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af",
            "asplit=2[low][high];[low]lowpass=f=2200[lowf];[high]highpass=f=2200,vibrato=f=4:d=0.04[highf];[lowf][highf]amix=inputs=2:duration=longest",
            str(output_path)
        ]
        kwargs = {"capture_output": True, "check": True}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.run(cmd, **kwargs)

    def _safe_peak(self, y: np.ndarray, ceiling: float = 0.85) -> np.ndarray:
        """Prevent intermediate WAV clipping before later processors/mastering."""
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > ceiling:
            return y / peak * ceiling
        return y

    def _sum_stems(self, stem_paths: dict[str, str]) -> np.ndarray:
        """Sum processed stems back into a single mix with safe gain staging (stereo-safe)."""
        total = None
        valid_stems = [path for path in stem_paths.values() if os.path.exists(path)]
        if not valid_stems:
            raise RuntimeError("No stems to sum")

        # Lower each stem before summing to avoid clipping
        stem_gain = 1.0 / max(1.0, np.sqrt(len(valid_stems)))

        for path in valid_stems:
            y, sr = load_audio(path, sr=self.sr, mono=False)
            y = self._safe_peak(y * stem_gain, ceiling=0.75)
            if total is None:
                total = y
            else:
                # Align lengths (axis=-1 is samples)
                max_len = max(total.shape[-1], y.shape[-1])
                # Match channel dimensions (ensure both are stereo)
                n_ch = max(total.shape[0], y.shape[0])
                
                # Helper to pad channels and samples
                def pad_audio(arr, chs, l):
                    # Pad channels if mono to make it stereo
                    if arr.shape[0] < chs:
                        arr = np.tile(arr, (chs, 1))
                    # Pad samples to match length
                    pad_len = l - arr.shape[-1]
                    if pad_len > 0:
                        arr = np.pad(arr, ((0, 0), (0, pad_len)), "constant")
                    return arr

                total = pad_audio(total, n_ch, max_len)
                y_padded = pad_audio(y, n_ch, max_len)
                total = total + y_padded

        return self._safe_peak(total, ceiling=0.85) if total is not None else total

    def remaster(self, input_path: str | Path, output_name: str | None = None,
                 progress_callback=None, reference_path: str | Path | None = None) -> RemasterResult:
        """Run the full remaster pipeline on an input file.

        Args:
            input_path: Path to the input audio file.
            output_name: Optional custom output filename (without path). If None,
                         uses 'remastered_{input_stem}.wav'.
            progress_callback: Optional callable that receives event dicts for progress tracking.
        """
        input_path = Path(input_path)
        notes: list[str] = []
        clean_path = ""
        mix_summed_path = self.output_dir / "mix_summed.wav"
        mix_path = ""
        phase_path = ""
        spatial_path = ""
        noise_path = ""
        master_path = ""
        pre_master_path = ""
        stems = None
        h_stems: dict[str, str] = {}
        s_stems: dict[str, str] = {}

        def _emit(step: str):
            print(f"[*] {Path(input_path).name} -> {step}", flush=True)
            if progress_callback:
                progress_callback({"type": "step", "step": step})

        # Step 1: Metadata purge
        try:
            self._check_cancel()
            _emit("Metadata purge...")
            notes.append("Step 1: Metadata purge...")
            clean_path = self._strip_metadata(input_path)

            # Step 2: Stem separation (requires torch)
            self._check_cancel()
            _emit("Stem separation (Demucs)...")
            notes.append("Step 2: Stem separation (Demucs)...")
            stems = self._get_stem_sep().separate(clean_path)
            notes.append(f"  Stems: v={stems.vocals_path}, d={stems.drums_path}, b={stems.bass_path}, o={stems.other_path}")

            # Step 3: Per-stem humanization
            self._check_cancel()
            _emit("Humanizing stems...")
            notes.append("Step 3: Humanization...")
            humanizer = self._get_humanizer()
            drift_cents = 8.0 if self.intensity == "aggressive" else 5.0
            shift_ms = 8.0 if self.intensity == "aggressive" else 5.0
            for stem_type, path in [
                ("vocals", stems.vocals_path),
                ("drums", stems.drums_path),
                ("bass", stems.bass_path),
                ("other", stems.other_path),
            ]:
                self._check_cancel()
                h_stems[stem_type] = humanizer.process_file(path, stem_type, drift_cents=drift_cents, shift_ms=shift_ms)
                notes.append(f"  {stem_type}: {h_stems[stem_type]}")

            # Step 4: Spectral saturation
            self._check_cancel()
            _emit("Spectral processing...")
            notes.append("Step 4: Spectral processing...")
            spectral = self._get_spectral()
            for stem_type, path in h_stems.items():
                self._check_cancel()
                s_stems[stem_type] = spectral.process_file(path, stem_type)
                notes.append(f"  {stem_type}: {s_stems[stem_type]}")

            # Step 5: Phase perturbation (on summed mix)
            self._check_cancel()
            _emit("Phase perturbation...")
            notes.append("Step 5: Phase perturbation...")
            mix = self._sum_stems(s_stems)
            # soundfile expects (samples, channels) — _sum_stems returns (channels, samples)
            mix_w = mix.T if mix.ndim > 1 else mix
            sf.write(str(mix_summed_path), mix_w, self.sr)
            mix_path = str(mix_summed_path)

            # Step 5.5: STFT spectral smoothing + high-freq vibrato (aggressive mode only)
            if self.intensity == "aggressive":
                self._check_cancel()
                _emit("Aggressive STFT smoothing...")
                notes.append("Step 5.5: STFT Spectral Smoothing...")
                from processors import SpectralPerturbator
                stft_proc = SpectralPerturbator(sr=self.sr)
                stft_path = stft_proc.process_file(mix_path, amount=0.15)
                mix_path = stft_path
                notes.append(f"  STFT smoothed: {stft_path}")
                
                self._check_cancel()
                # High-freq vibrato (>2kHz) to break checkerboard without touching vocals
                hf_path = str(self.output_dir / "mix_hf_vib.wav")
                try:
                    self._ffmpeg_highfreq_vibrato(mix_path, hf_path)
                    mix_path = hf_path
                    notes.append(f"  HF Vibrato: {hf_path}")
                except Exception:
                    notes.append("  FFmpeg not available, skipping")

            self._check_cancel()
            phase = self._get_phase()
            phase_amount = 0.08 if self.intensity == "aggressive" else 0.05
            phase_path = phase.process_file(str(mix_path), amount=phase_amount)
            notes.append(f"  Phase: {phase_path}")

            # Step 6: Spatial (reverb) + Noise + Master
            self._check_cancel()
            _emit("Spatial, noise, and mastering...")
            notes.append("Step 6: Spatial + Noise + Master...")
            spatial = self._get_spatial()
            spatial_path = spatial.process_file(phase_path, mix=0.08)
            notes.append(f"  Spatial: {spatial_path}")

            self._check_cancel()
            noise = self._get_noise()
            noise_path = noise.process_file(spatial_path, level_db=-66)
            notes.append(f"  Noise: {noise_path}")

            # Step 6.5: Reference Track Matching (optional)
            pre_master_path = noise_path
            if reference_path and os.path.exists(str(reference_path)):
                self._check_cancel()
                _emit("Reference matching...")
                notes.append("Step 6.5: Reference Track Matching...")
                from processors.master import ReferenceMaster
                ref_master = ReferenceMaster(sr=self.sr)
                ref_matched_path = ref_master.process_file(noise_path, reference_path)
                pre_master_path = ref_matched_path
                notes.append(f"  Reference matched: {ref_matched_path}")

            self._check_cancel()
            master = self._get_master()
            out_name = output_name or f"remastered_{input_path.stem}.wav"
            out_path = self.output_dir / out_name
            
            # Pyloudnorm & master chain processes (samples, channels) internally.
            master_path = master.process_file(pre_master_path)
            shutil.copy2(master_path, out_path)
            notes.append(f"  Master: {out_path}")
        finally:
            # Cleanup intermediate files regardless of success or cancellation
            out_path_str = str(out_path) if 'out_path' in locals() else ""
            
            stem_paths = []
            if stems:
                stem_paths = [stems.vocals_path, stems.drums_path, stems.bass_path, stems.other_path]
                stem_paths.extend(getattr(stems, "extra_paths", []))
                
            for p in [clean_path, str(mix_summed_path), mix_path, phase_path, spatial_path, noise_path, master_path,
                      str(pre_master_path), *stem_paths, *h_stems.values(), *s_stems.values()]:
                try:
                    if p and os.path.exists(p) and p != out_path_str:
                        os.remove(p)
                except Exception:
                    pass

        if self.shutdown_event and self.shutdown_event.is_set():
            raise CancellationError("Pipeline aborted gracefully.")
            
        notes.append("  Pipeline complete!")
        return RemasterResult(
            filepath=str(out_path),
            original_path=str(input_path),
            notes=notes,
        )

    def batch_remaster(self, folder_path: str | Path, progress_callback=None,
                       reference_path: str | Path | None = None) -> list[RemasterResult]:
        """Remaster all audio files in a folder.

        Scans folder for .mp3, .wav, .flac, .ogg, .m4a files.
        Output goes to folder/output/ subdirectory, preserving original filenames
        with .wav extension.

        Args:
            folder_path: Path to folder containing audio files.

        Returns:
            List of RemasterResult for each successfully processed file.
        """
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a folder: {folder_path}")

        audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
        audio_files = sorted(
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in audio_extensions
        )

        if not audio_files:
            raise FileNotFoundError(f"No audio files found in: {folder_path}")

        output_subdir = folder_path / "output"
        output_subdir.mkdir(exist_ok=True)

        # Temporarily override output_dir to the folder's output/ subdirectory
        original_output_dir = self.output_dir
        self.output_dir = output_subdir

        results: list[RemasterResult] = []
        try:
            for i, audio_file in enumerate(audio_files, 1):
                output_name = f"{audio_file.stem}.wav"
                print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")
                if progress_callback:
                    progress_callback({
                        "type": "file_start",
                        "index": i,
                        "total": len(audio_files),
                        "filename": audio_file.name,
                        "input_path": str(audio_file),
                    })

                try:
                    def _file_progress(event: dict):
                        if progress_callback:
                            progress_callback({
                                **event,
                                "index": i,
                                "total": len(audio_files),
                                "filename": audio_file.name,
                                "input_path": str(audio_file),
                            })

                    result = self.remaster(str(audio_file), output_name=output_name,
                                           progress_callback=_file_progress, reference_path=reference_path)
                    results.append(result)
                    print(f"  ✅ Done: {output_name}")
                    if progress_callback:
                        progress_callback({
                            "type": "file_remastered",
                            "index": i,
                            "total": len(audio_files),
                            "filename": audio_file.name,
                            "input_path": str(audio_file),
                            "output_path": result.filepath,
                        })
                except CancellationError as e:
                    print(f"  🛑 Aborted (cancellation): {e}")
                    result = RemasterResult(
                        filepath="",
                        original_path=str(audio_file),
                        notes=[f"Cancelled: {e}"],
                    )
                    results.append(result)
                    if progress_callback:
                        progress_callback({
                            "type": "file_failed",
                            "index": i,
                            "total": len(audio_files),
                            "filename": audio_file.name,
                            "input_path": str(audio_file),
                            "error": "Cancelled",
                        })
                    break  # Stop processing remaining files in batch
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    result = RemasterResult(
                        filepath="",
                        original_path=str(audio_file),
                        notes=[f"Error: {e}"],
                    )
                    results.append(result)
                    if progress_callback:
                        progress_callback({
                            "type": "file_failed",
                            "index": i,
                            "total": len(audio_files),
                            "filename": audio_file.name,
                            "input_path": str(audio_file),
                            "error": str(e),
                        })
                    
        finally:
            self.output_dir = original_output_dir

        return results
