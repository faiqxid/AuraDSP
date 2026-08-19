"""Spectral Analyzer — Detect AI spectral fingerprints (Checkerboard Artifacts).

AI music generators (Suno, Udio, dll.) pakai deconvolution layers yang menghasilkan
systematic spectral peaks di predictable frequency intervals — "checkerboard artifacts".
ISMR 2025 paper proved logistic regression 10K params can detect this at 99%+ accuracy.

Strategy: analyse spectral flatness, centroid consistency, rolloff patterns,
harmonic-to-noise ratio, dan spectral flux regularity.
"""
import numpy as np
import librosa
from dataclasses import dataclass


@dataclass
class SpectralResult:
    score: float  # 0.0 = definitely human, 1.0 = definitely AI
    confidence: str  # "low", "medium", "high"
    details: dict
    verdict: str  # "human", "ai", "uncertain"


class SpectralAnalyzer:
    """Analyze spectral characteristics for AI detection markers."""

    # AI audio tends to have unnaturally consistent spectral features over time.
    # Human music has more variance from room acoustics, mic bleed, performance dynamics.

    # Thresholds (calibrated heuristics — will need tuning with real datasets)
    SPECTRAL_FLATNESS_VARIANCE_LOW = 0.002    # AI: very low variance in spectral flatness
    SPECTRAL_FLATNESS_VARIANCE_HIGH = 0.015   # Human: higher variance
    SPECTRAL_CENTROID_CONSISTENCY_LOW = 50.0   # AI: centroid std dev across windows (Hz)
    SPECTRAL_CENTROID_CONSISTENCY_HIGH = 200.0 # Human
    HNR_HIGH = 25.0   # AI: higher harmonic-to-noise ratio (too clean)
    HNR_LOW = 10.0    # Human: more noise floor
    FLUX_REGULARITY_LOW = 0.15   # AI: spectral flux coefficient of variation
    FLUX_REGULARITY_HIGH = 0.45  # Human: more irregular flux

    def analyze(self, y: np.ndarray, sr: int) -> SpectralResult:
        """Run full spectral analysis on audio signal."""
        # STFT parameters
        hop_length = 512
        n_fft = 2048

        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        # --- Feature 1: Spectral Flatness Variance ---
        # AI audio has unnaturally uniform spectral flatness across time
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        flatness_var = float(np.var(flatness))
        flatness_mean = float(np.mean(flatness))

        # --- Feature 2: Spectral Centroid Consistency ---
        # AI: centroid stays in tight range; Human: more drift
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        centroid_std = float(np.std(centroid))
        centroid_mean = float(np.mean(centroid))

        # --- Feature 3: Spectral Rolloff Pattern ---
        # AI generators produce predictable rolloff curves
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
        rolloff_std = float(np.std(rolloff))
        rolloff_ratio = rolloff_std / (centroid_mean + 1e-8)  # Normalized

        # --- Feature 4: Spectral Flux Regularity ---
        # AI: flux changes are more periodic/regular; Human: more organic
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        flux_cv = float(np.std(flux) / (np.mean(flux) + 1e-8))  # Coefficient of variation

        # --- Feature 5: Harmonic-to-Noise Ratio ---
        # AI audio tends to be "too clean" — higher HNR
        harmonic, percussive = librosa.effects.hpss(y)
        h_noise = y - harmonic
        h_power = np.mean(harmonic ** 2)
        n_power = np.mean(h_noise ** 2) + 1e-10
        hnr = float(10 * np.log10(h_power / n_power))

        # --- Feature 6: Spectral Bandwidth Consistency ---
        # AI: bandwidth stays in narrow range
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        bandwidth_cv = float(np.std(bandwidth) / (np.mean(bandwidth) + 1e-8))

        # --- Feature 7: Spectral Contrast Regularity ---
        # Checkerboard artifacts produce regular contrast patterns
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        contrast_var = float(np.mean(np.var(contrast, axis=1)))

        # --- Scoring ---
        scores = []

        # Flatness variance: low = AI
        s = self._score_range(flatness_var, self.SPECTRAL_FLATNESS_VARIANCE_LOW, self.SPECTRAL_FLATNESS_VARIANCE_HIGH)
        scores.append(("spectral_flatness_variance", s, 1.5))

        # Centroid consistency: low std = AI
        s = self._score_range(centroid_std, self.SPECTRAL_CENTROID_CONSISTENCY_LOW, self.SPECTRAL_CENTROID_CONSISTENCY_HIGH)
        scores.append(("spectral_centroid_consistency", s, 1.5))

        # HNR: high = AI (too clean)
        s = self._score_range(hnr, self.HNR_LOW, self.HNR_HIGH, inverted=True)
        scores.append(("harmonic_noise_ratio", s, 1.0))

        # Flux regularity: low CV = AI (too regular)
        s = self._score_range(flux_cv, self.FLUX_REGULARITY_LOW, self.FLUX_REGULARITY_HIGH)
        scores.append(("spectral_flux_regularity", s, 1.0))

        # Bandwidth consistency: low CV = AI
        # Typical range 0.05 (AI) to 0.3 (human)
        s = self._score_range(bandwidth_cv, 0.05, 0.3)
        scores.append(("bandwidth_consistency", s, 0.8))

        # Contrast regularity: low = AI
        # Typical range 30 (AI) to 80 (human)
        s = self._score_range(contrast_var, 30.0, 80.0)
        scores.append(("contrast_regularity", s, 0.7))

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

        return SpectralResult(
            score=round(final_score, 4),
            confidence=confidence,
            verdict=verdict,
            details={
                "spectral_flatness_variance": round(flatness_var, 6),
                "spectral_flatness_mean": round(flatness_mean, 6),
                "spectral_centroid_std_hz": round(centroid_std, 2),
                "spectral_centroid_mean_hz": round(centroid_mean, 2),
                "spectral_rolloff_ratio": round(rolloff_ratio, 4),
                "spectral_flux_cv": round(flux_cv, 4),
                "harmonic_noise_ratio_db": round(hnr, 2),
                "bandwidth_cv": round(bandwidth_cv, 4),
                "spectral_contrast_variance": round(contrast_var, 4),
                "sub_scores": {name: round(s, 4) for name, s, _ in scores},
            },
        )

    @staticmethod
    def _score_range(value: float, low: float, high: float, inverted: bool = False) -> float:
        """Map value to 0-1 AI score. low=AI, high=human (unless inverted: low=human, high=AI)."""
        if high == low:
            return 0.5
        normalized = (value - low) / (high - low)
        normalized = max(0.0, min(1.0, normalized))
        # low=AI -> high AI score at low value
        score = 1.0 - normalized
        if inverted:
            score = normalized
        return score
