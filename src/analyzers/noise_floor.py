"""Noise Floor Analyzer — Detect artificially clean or patterned noise floors.

AI-generated audio has noise floor characteristics that differ from real recordings:
- Too clean (SNR unnaturally high)
- Noise has statistical regularity (Gaussian, uniform) vs organic noise
- No room tone / ambient character
- No mic self-noise / preamp noise signatures
"""
import numpy as np
import scipy.stats as stats
from dataclasses import dataclass


@dataclass
class NoiseFloorResult:
    score: float  # 0.0 = definitely human, 1.0 = definitely AI
    confidence: str  # "low", "medium", "high"
    details: dict
    verdict: str  # "human", "ai", "uncertain"


class NoiseFloorAnalyzer:
    """Analyze noise floor characteristics for AI detection."""

    # Thresholds
    SNR_HIGH = 70.0   # AI: unnaturally high SNR
    SNR_LOW = 30.0     # Human: real recordings have lower SNR
    KURTOSIS_LOW = 2.5   # AI: noise is more Gaussian (kurtosis ≈ 3)
    KURTOSIS_HIGH = 8.0  # Human: noise has heavier tails (impulse noise, etc.)
    SPECTRAL_FLATNESS_LOW = 0.3   # AI: noise floor spectral shape is more flat/uniform
    SPECTRAL_FLATNESS_HIGH = 0.8  # Human: noise floor has more spectral character

    def analyze(self, y: np.ndarray, sr: int) -> NoiseFloorResult:
        """Analyze noise floor characteristics."""

        # --- Estimate noise floor ---
        # Method: take the quietest 10% of frames as noise estimate
        frame_length = 2048
        hop = 512
        n_frames = 1 + (len(y) - frame_length) // hop
        if n_frames < 10:
            return NoiseFloorResult(
                score=0.5, confidence="low", verdict="uncertain",
                details={"error": "Audio too short for noise floor analysis"},
            )

        frames = np.lib.stride_tricks.as_strided(
            y,
            shape=(n_frames, frame_length),
            strides=(y.strides[0] * hop, y.strides[0]),
        )
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

        # Quietest 10% = noise floor estimate
        threshold = np.percentile(frame_rms, 10)
        noise_mask = frame_rms <= threshold
        noise_frames = frames[noise_mask]

        if len(noise_frames) < 5:
            return NoiseFloorResult(
                score=0.5, confidence="low", verdict="uncertain",
                details={"error": "Not enough quiet frames for noise analysis"},
            )

        noise_signal = noise_frames.flatten()

        # --- Feature 1: Signal-to-Noise Ratio ---
        signal_rms = np.sqrt(np.mean(y ** 2))
        noise_rms = np.sqrt(np.mean(noise_signal ** 2)) + 1e-10
        snr_db = float(20 * np.log10(signal_rms / noise_rms))

        # --- Feature 2: Noise Kurtosis ---
        # AI noise is more Gaussian (kurtosis ≈ 3). Human noise has heavier tails.
        noise_kurtosis = float(stats.kurtosis(noise_signal))

        # --- Feature 3: Noise Skewness ---
        # AI noise is more symmetric. Human noise can be skewed.
        noise_skewness = float(abs(stats.skew(noise_signal)))

        # --- Feature 4: Noise Spectral Flatness ---
        # AI noise is spectrally uniform. Human noise has spectral character.
        import librosa
        noise_stft = np.abs(np.fft.rfft(noise_signal[:4096]))
        if len(noise_stft) > 1:
            geo_mean = np.exp(np.mean(np.log(noise_stft + 1e-10)))
            arith_mean = np.mean(noise_stft) + 1e-10
            noise_spectral_flatness = float(geo_mean / arith_mean)
        else:
            noise_spectral_flatness = 0.5

        # --- Feature 5: Noise Autocorrelation ---
        # AI noise has more predictable autocorrelation structure
        if len(noise_signal) > 1000:
            sample = noise_signal[:4096]
            autocorr = np.correlate(sample, sample, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            # How fast autocorrelation decays
            autocorr_decay = float(np.mean(np.abs(np.diff(autocorr[:20]))))
        else:
            autocorr_decay = 0.5

        # --- Feature 6: Noise Floor Stability ---
        # AI noise floor is more stable (less variation across the track)
        noise_floor_over_time = frame_rms[noise_mask]
        if len(noise_floor_over_time) > 2:
            nf_stability = float(np.std(noise_floor_over_time) / (np.mean(noise_floor_over_time) + 1e-10))
        else:
            nf_stability = 0.5

        # --- Scoring ---
        scores = []

        # High SNR = AI (too clean)
        s = self._score_range(snr_db, self.SNR_LOW, self.SNR_HIGH)
        scores.append(("snr_analysis", s, 1.5))

        # Kurtosis near 3 (Gaussian) = AI
        kurtosis_score = 1.0 - abs(noise_kurtosis - 3.0) / 10.0
        kurtosis_score = max(0.0, min(1.0, kurtosis_score))
        scores.append(("noise_kurtosis", kurtosis_score, 1.5))

        # Low skewness = more Gaussian = AI
        skewness_score = 1.0 - noise_skewness
        skewness_score = max(0.0, min(1.0, skewness_score))
        scores.append(("noise_skewness", skewness_score, 0.8))

        # Spectral flatness of noise: flat = AI
        s = self._score_range(noise_spectral_flatness, self.SPECTRAL_FLATNESS_LOW, self.SPECTRAL_FLATNESS_HIGH)
        scores.append(("noise_spectral_flatness", s, 1.0))

        # Fast autocorrelation decay = AI (noise is more independent)
        # Typical: AI 0.05-0.15, Human 0.02-0.08
        s = self._score_range(autocorr_decay, 0.03, 0.12)
        scores.append(("autocorr_decay", s, 0.7))

        # Noise floor stability: high stability = AI
        s = self._score_range(nf_stability, 0.05, 0.25)
        scores.append(("noise_floor_stability", s, 0.8))

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

        return NoiseFloorResult(
            score=round(final_score, 4),
            confidence=confidence,
            verdict=verdict,
            details={
                "snr_db": round(snr_db, 2),
                "noise_kurtosis": round(noise_kurtosis, 4),
                "noise_skewness": round(noise_skewness, 4),
                "noise_spectral_flatness": round(noise_spectral_flatness, 4),
                "autocorr_decay": round(autocorr_decay, 4),
                "noise_floor_stability": round(nf_stability, 4),
                "sub_scores": {name: round(s, 4) for name, s, _ in scores},
            },
        )

    @staticmethod
    def _score_range(value: float, low: float, high: float) -> float:
        """Map value to 0-1 score. low=AI, high=human (depends on context)."""
        if high == low:
            return 0.5
        score = (value - low) / (high - low)
        return max(0.0, min(1.0, score))
