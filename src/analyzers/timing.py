"""Timing Analyzer — Detect unnaturally precise timing in AI-generated music.

AI music generators snap transients to a mathematically perfect grid.
Human musicians always introduce micro-timing variations (typically ±5-15ms)
following gaussian-like distributions. IBI (Inter-Beat Interval) variance
approaches zero in AI music.

Detection approach:
1. Onset detection — find note/beat transients
2. IBI variance — measure regularity of inter-onset intervals
3. Onset strength regularity — AI has more uniform onset magnitudes
4. Micro-timing analysis — measure deviations from grid (should be near-zero for AI)
"""
import numpy as np
import librosa
from dataclasses import dataclass


@dataclass
class TimingResult:
    score: float  # 0.0 = definitely human, 1.0 = definitely AI
    confidence: str  # "low", "medium", "high"
    details: dict
    verdict: str  # "human", "ai", "uncertain"


class TimingAnalyzer:
    """Analyze timing characteristics for AI detection."""

    # Thresholds
    IBI_CV_LOW = 0.008     # AI: very low coefficient of variation in inter-beat intervals
    IBI_CV_HIGH = 0.06     # Human: higher variation
    ONSET_STRENGTH_CV_LOW = 0.15   # AI: uniform onset strengths
    ONSET_STRENGTH_CV_HIGH = 0.50  # Human: varied dynamics
    GRID_DEVIATION_LOW = 1.0    # AI: onset timing deviates < ~1ms from perfect grid
    GRID_DEVIATION_HIGH = 12.0  # Human: onset timing deviates ~5-15ms

    def analyze(self, y: np.ndarray, sr: int) -> TimingResult:
        """Run timing analysis on audio signal."""

        # --- Feature 1: Inter-Beat Interval (IBI) Variance ---
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beats, sr=sr)

        if len(beat_times) < 4:
            return TimingResult(
                score=0.5, confidence="low", verdict="uncertain",
                details={"error": "Too few beats detected for timing analysis"},
            )

        ibi = np.diff(beat_times)  # Inter-beat intervals
        ibi_mean = float(np.mean(ibi))
        ibi_std = float(np.std(ibi))
        ibi_cv = ibi_std / (ibi_mean + 1e-10)  # Coefficient of variation

        # --- Feature 2: Onset Strength Regularity ---
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_cv = float(np.std(onset_env) / (np.mean(onset_env) + 1e-10))

        # --- Feature 3: Onset Detection + Grid Deviation ---
        onsets = librosa.onset.onset_detect(
            y=y, sr=sr, units="time",
            backtrack=False,
            delta=0.07,
        )

        if len(onsets) < 8:
            grid_deviation_ms = 5.0  # Neutral
        else:
            # Estimate expected grid from median inter-onset interval
            ioi = np.diff(onsets)
            median_ioi = float(np.median(ioi))
            if median_ioi > 0:
                # How far each onset is from the nearest grid point
                expected_grid = np.arange(onsets[0], onsets[-1], median_ioi)
                deviations = []
                for o in onsets:
                    nearest = expected_grid[np.argmin(np.abs(expected_grid - o))]
                    deviations.append(abs(o - nearest))
                grid_deviation_ms = float(np.median(deviations) * 1000)
            else:
                grid_deviation_ms = 5.0

        # --- Feature 4: Beat Phase Coherence ---
        # AI beats have near-zero phase jitter
        if ibi_mean > 0:
            phase_jitter = float(np.std(ibi) / ibi_mean)
        else:
            phase_jitter = 0.5

        # --- Feature 5: Dynamics Consistency ---
        # Measure RMS envelope variance — AI is more consistent
        frame_length = 2048
        hop = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
        dynamics_cv = float(np.std(rms) / (np.mean(rms) + 1e-10))

        # --- Scoring ---
        scores = []

        s = self._score_range(ibi_cv, self.IBI_CV_LOW, self.IBI_CV_HIGH)
        scores.append(("ibi_coefficient_of_variation", s, 2.0))

        s = self._score_range(onset_cv, self.ONSET_STRENGTH_CV_LOW, self.ONSET_STRENGTH_CV_HIGH)
        scores.append(("onset_strength_regularity", s, 1.5))

        s = self._score_range(grid_deviation_ms, self.GRID_DEVIATION_LOW, self.GRID_DEVIATION_HIGH)
        scores.append(("grid_deviation_ms", s, 2.0))

        # Phase jitter
        s = self._score_range(phase_jitter, 0.005, 0.05)
        scores.append(("beat_phase_jitter", s, 1.0))

        # Dynamics: high CV = human (varied dynamics)
        s = self._score_range(dynamics_cv, 0.3, 1.2)
        scores.append(("dynamics_consistency", s, 0.8))

        # Weighted average
        total_weight = sum(w for _, _, w in scores)
        final_score = sum(s * w for _, s, w in scores) / total_weight
        final_score = max(0.0, min(1.0, final_score))

        # Verdict
        if final_score >= 0.65:
            verdict = "ai"
            confidence = "high" if final_score >= 0.80 else "medium"
        elif final_score <= 0.35:
            verdict = "human"
            confidence = "high" if final_score <= 0.20 else "medium"
        else:
            verdict = "uncertain"
            confidence = "low"

        return TimingResult(
            score=round(final_score, 4),
            confidence=confidence,
            verdict=verdict,
            details={
                "tempo_bpm": round(float(tempo), 2) if isinstance(tempo, (int, float)) else round(float(tempo[0]), 2),
                "beats_detected": len(beat_times),
                "ibi_mean_sec": round(ibi_mean, 6),
                "ibi_std_sec": round(ibi_std, 6),
                "ibi_cv": round(ibi_cv, 6),
                "onset_strength_cv": round(onset_cv, 4),
                "grid_deviation_ms": round(grid_deviation_ms, 2),
                "beat_phase_jitter": round(phase_jitter, 6),
                "dynamics_cv": round(dynamics_cv, 4),
                "sub_scores": {name: round(s, 4) for name, s, _ in scores},
            },
        )

    @staticmethod
    def _score_range(value: float, low: float, high: float) -> float:
        """Map value to 0-1 AI score. low=AI, high=human."""
        if high == low:
            return 0.5
        normalized = (value - low) / (high - low)
        normalized = max(0.0, min(1.0, normalized))
        return 1.0 - normalized
