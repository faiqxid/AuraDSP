"""AI Music Detector — SOTA model primary + heuristic secondary.

Primary engine: lofcz/ai-music-detector (logistic regression, 99.88% accuracy)
  - Input: spectral fakeprint (3585 features)
  - Targets: Suno <= 5, Udio <= 1.5
  
Secondary engine: 5-vector heuristic analysis (spectral, timing, phase, noise, metadata)
  - Provides explanatory breakdown per detection vector
  - Does NOT drive the final verdict — that's the SOTA model's job
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from analyzers.fakeprint import extract_fakeprint, PreprocessingConfig
from analyzers.sota_model import SOTADetector
from analyzers.spectral import SpectralAnalyzer
from analyzers.phase import PhaseAnalyzer
from analyzers.timing import TimingAnalyzer
from analyzers.noise_floor import NoiseFloorAnalyzer
from analyzers.metadata import MetadataAnalyzer
from utils.audio_io import get_audio_info, load_audio


@dataclass
class DetectorResult:
    filepath: str
    # SOTA model primary score
    score: float
    percentage: float
    verdict: str
    confidence: str
    # Heuristic secondary scores (for explanation)
    vector_scores: dict[str, float]
    vector_verdicts: dict[str, str]
    audio_info: dict[str, Any]
    details: dict[str, Any]
    notes: list[str]


class AIMusicDetector:
    """Hybrid AI music detector: SOTA model primary + heuristic explanation."""

    def __init__(self, sample_rate: int = 22050, max_seconds: float | None = 180.0):
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        # Primary: SOTA model
        self.sota = SOTADetector()
        self.prep_config = PreprocessingConfig()
        # Secondary: heuristic analyzers
        self.spectral = SpectralAnalyzer()
        self.phase = PhaseAnalyzer()
        self.timing = TimingAnalyzer()
        self.noise_floor = NoiseFloorAnalyzer()
        self.metadata = MetadataAnalyzer()

    def detect(self, filepath: str) -> DetectorResult:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(filepath)

        print(f"[*] Analyzing fingerprints: {path.name}", flush=True)
        info = get_audio_info(filepath)

        # === PRIMARY: SOTA model ===
        fakeprint = extract_fakeprint(str(path), self.prep_config)
        sota_result = self.sota.predict(fakeprint)

        # === SECONDARY: Heuristic analysis ===
        y, sr = load_audio(filepath, sr=self.sample_rate, mono=True)

        if self.max_seconds and len(y) > int(sr * self.max_seconds):
            y = y[: int(sr * self.max_seconds)]

        peak = abs(y).max() if len(y) else 0
        if peak > 0:
            y = y / peak * 0.95

        heuristic_results = {
            "spectral": self.spectral.analyze(y, sr),
            "timing": self.timing.analyze(y, sr),
            "phase": self.phase.analyze(y, sr),
            "noise_floor": self.noise_floor.analyze(y, sr),
            "metadata": self.metadata.analyze(filepath),
        }

        vector_scores = {k: float(v.score) for k, v in heuristic_results.items()}
        vector_verdicts = {k: v.verdict for k, v in heuristic_results.items()}

        # === COMBINE: SOTA verdict is authoritative ===
        final_score = sota_result.score
        verdict = sota_result.verdict.upper()
        confidence = sota_result.confidence

        # Map to richer verdict labels
        if verdict == "AI":
            if final_score >= 0.90:
                verdict = "AI_GENERATED"
            else:
                verdict = "LIKELY_AI"
        elif verdict == "HUMAN":
            if final_score <= 0.10:
                verdict = "HUMAN"
            else:
                verdict = "LIKELY_HUMAN"

        notes = self._notes(final_score, vector_scores, vector_verdicts, sota_result)

        return DetectorResult(
            filepath=str(path.absolute()),
            score=round(final_score, 6),
            percentage=round(final_score * 100, 2),
            verdict=verdict,
            confidence=confidence,
            audio_info=asdict(info),
            vector_scores={k: round(v, 4) for k, v in vector_scores.items()},
            vector_verdicts=vector_verdicts,
            details={
                "sota_model": sota_result.details,
                "heuristic": {k: v.details for k, v in heuristic_results.items()},
            },
            notes=notes,
        )

    def _notes(
        self,
        score: float,
        vector_scores: dict[str, float],
        vector_verdicts: dict[str, str],
        sota,
    ) -> list[str]:
        notes: list[str] = []

        # SOTA model note
        notes.append(
            f"SOTA model (lofcz/ai-music-detector): {score*100:.1f}% AI probability "
            f"— targets Suno <= 5, Udio <= 1.5. Accuracy: 99.88%."
        )

        # Heuristic observations
        if vector_scores.get("metadata", 0) >= 0.5:
            notes.append("AI-related metadata/watermark markers found.")
        if vector_scores.get("spectral", 0) >= 0.65:
            notes.append("Spectral features look unusually regular, consistent with vocoder artifacts.")
        if vector_scores.get("timing", 0) >= 0.65:
            notes.append("Timing/onset grid appears unusually precise.")
        if vector_scores.get("phase", 0) >= 0.65:
            notes.append("Phase coherence looks synthetic or over-regularized.")
        if vector_scores.get("noise_floor", 0) >= 0.65:
            notes.append("Noise floor looks artificial or too clean.")

        if not any(v >= 0.5 for v in vector_scores.values()):
            notes.append("No heuristic vector strongly flags AI — model score drives verdict.")

        return notes
