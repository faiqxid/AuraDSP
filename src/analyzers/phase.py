"""Phase Analyzer — Detect anomalous phase coherence in AI-generated audio.

AI audio has anomalously low phase entropy — phase relationships are "too perfect".
Detection uses Hilbert Transform to extract instantaneous phase, then computes
Shannon Entropy on the phase distribution.

Human recordings have high phase entropy (naturally chaotic phase relationships).
AI generators produce impossibly perfect phase locking between frequency components.
"""
import numpy as np
import scipy.signal as signal
from dataclasses import dataclass


@dataclass
class PhaseResult:
    score: float  # 0.0 = definitely human, 1.0 = definitely AI
    confidence: str  # "low", "medium", "high"
    details: dict
    verdict: str  # "human", "ai", "uncertain"


class PhaseAnalyzer:
    """Analyze phase coherence characteristics for AI detection."""

    # Thresholds
    PHASE_ENTROPY_LOW = 2.5    # AI: low entropy (too coherent)
    PHASE_ENTROPY_HIGH = 4.0   # Human: high entropy (natural chaos)
    PHASE_VARIANCE_LOW = 0.05  # AI: very low phase variance
    PHASE_VARIANCE_HIGH = 0.35 # Human: higher phase variance

    def analyze(self, y: np.ndarray, sr: int) -> PhaseResult:
        """Run phase coherence analysis on audio signal."""
        # Split into frequency bands for per-band analysis
        bands = [
            ("sub_bass", 20, 80),
            ("bass", 80, 250),
            ("low_mid", 250, 800),
            ("mid", 800, 2500),
            ("high_mid", 2500, 6000),
            ("high", 6000, 12000),
            ("brilliance", 12000, 20000),
        ]

        band_results = {}
        band_entropies = []
        band_variances = []

        nyquist = sr / 2
        for name, low, high in bands:
            if high >= nyquist:
                high = nyquist * 0.95
            if low >= nyquist:
                continue

            # Bandpass filter
            low_norm = low / nyquist
            high_norm = high / nyquist
            if low_norm <= 0 or high_norm >= 1 or low_norm >= high_norm:
                continue

            try:
                b, a = signal.butter(4, [low_norm, high_norm], btype="band")
                band_signal = signal.filtfilt(b, a, y)
            except Exception:
                continue

            # Skip if signal is too quiet
            if np.max(np.abs(band_signal)) < 1e-6:
                continue

            # Hilbert transform to get analytic signal
            analytic = signal.hilbert(band_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic))

            # Clean NaN/Inf
            instantaneous_phase = instantaneous_phase[np.isfinite(instantaneous_phase)]
            if len(instantaneous_phase) < 64:
                continue

            # Phase entropy (Shannon entropy of phase distribution)
            phase_hist, _ = np.histogram(
                instantaneous_phase % (2 * np.pi),
                bins=64,
                density=True,
            )
            phase_hist = phase_hist[phase_hist > 0]
            entropy = -np.sum(phase_hist * np.log2(phase_hist + 1e-10))
            entropy = float(entropy / np.log2(64))  # Normalize to 0-1

            # Phase variance (normalized)
            phase_diff = np.diff(instantaneous_phase)
            phase_var = float(np.var(phase_diff))

            # Phase autocorrelation — AI has higher autocorrelation (more periodic)
            if len(phase_diff) > 100:
                autocorr = np.correlate(phase_diff[:1000], phase_diff[:1000], mode="full")
                autocorr = autocorr[len(autocorr) // 2:]
                autocorr = autocorr / (autocorr[0] + 1e-10)
                # How fast autocorrelation drops — AI drops slower
                decay_rate = float(np.mean(np.abs(np.diff(autocorr[:50]))))
            else:
                decay_rate = 0.5  # Neutral

            band_entropies.append(entropy)
            band_variances.append(phase_var)
            band_results[name] = {
                "phase_entropy": round(entropy, 4),
                "phase_variance": round(phase_var, 6),
                "autocorr_decay": round(decay_rate, 4),
            }

        if not band_entropies:
            return PhaseResult(
                score=0.5, confidence="low", verdict="uncertain",
                details={"error": "No valid frequency bands for analysis"},
            )

        # Aggregate metrics
        mean_entropy = float(np.mean(band_entropies))
        mean_variance = float(np.mean(band_variances))

        # Score: low entropy = AI
        entropy_score = self._score_range(mean_entropy, self.PHASE_ENTROPY_LOW, self.PHASE_ENTROPY_HIGH)
        variance_score = self._score_range(mean_variance, self.PHASE_VARIANCE_LOW, self.PHASE_VARIANCE_HIGH)

        # High-freq bands are more diagnostic (AI artifacts are stronger there)
        high_band_scores = []
        for name in ["high_mid", "high", "brilliance"]:
            if name in band_results:
                e = band_results[name]["phase_entropy"]
                high_band_scores.append(self._score_range(e, self.PHASE_ENTROPY_LOW, self.PHASE_ENTROPY_HIGH))

        if high_band_scores:
            high_freq_score = float(np.mean(high_band_scores))
        else:
            high_freq_score = entropy_score

        # Weighted combination
        final_score = entropy_score * 0.35 + variance_score * 0.25 + high_freq_score * 0.40
        final_score = max(0.0, min(1.0, final_score))

        # Verdict
        if final_score >= 0.60:
            verdict = "ai"
            confidence = "high" if final_score >= 0.75 else "medium"
        elif final_score <= 0.35:
            verdict = "human"
            confidence = "high" if final_score <= 0.20 else "medium"
        else:
            verdict = "uncertain"
            confidence = "low"

        return PhaseResult(
            score=round(final_score, 4),
            confidence=confidence,
            verdict=verdict,
            details={
                "mean_phase_entropy": round(mean_entropy, 4),
                "mean_phase_variance": round(mean_variance, 6),
                "entropy_score": round(entropy_score, 4),
                "variance_score": round(variance_score, 4),
                "high_freq_score": round(high_freq_score, 4),
                "bands_analyzed": len(band_entropies),
                "band_details": band_results,
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
