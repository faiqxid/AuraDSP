"""SOTA Model Detector — lofcz/ai-music-detector integration.

This loads the pre-trained logistic regression model from HuggingFace
and runs inference on fakeprint feature vectors.

Model: logistic_regression (3585 features → sigmoid → AI probability)
Accuracy: 99.88% on 17,866 test samples (5,741 real, 12,125 AI)
Target: Suno <= 5, Udio <= 1.5
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SOTAResult:
    score: float       # 0.0 = definitely human, 1.0 = definitely AI
    confidence: str    # "low", "medium", "high"
    verdict: str       # "human", "ai", "uncertain"
    details: dict


class SOTADetector:
    """Pre-trained AI music detector using lofcz/ai-music-detector model."""

    def __init__(self, model_dir: str | None = None):
        if model_dir is None:
            from utils.frozen import get_base_dir
            model_dir = str(get_base_dir() / "models")
        
        self.model_dir = Path(model_dir)
        self.weights = None
        self.bias = None
        self._load_model()

    def _load_model(self):
        """Load model weights from safetensors file."""
        safetensors_path = self.model_dir / "model.safetensors"
        
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"Model not found at {safetensors_path}. "
                "Download from: https://huggingface.co/lofcz/ai-music-detector"
            )

        try:
            from safetensors.numpy import load_file
            weights_data = load_file(str(safetensors_path))
            self.weights = weights_data["weights"]  # Shape: (1, 3585)
            self.bias = weights_data["bias"]         # Shape: (1,)
        except ImportError:
            # Fallback: try loading via numpy if safetensors not available
            # This won't work for safetensors format, but worth trying
            raise ImportError(
                "Install safetensors: pip install safetensors\n"
                "Or download model.safetensors from huggingface"
            )

    def predict(self, fakeprint: np.ndarray) -> SOTAResult:
        """Run inference on a fakeprint feature vector.

        Args:
            fakeprint: numpy array of shape (3585,)

        Returns:
            SOTAResult with AI probability score
        """
        if self.weights is None:
            return SOTAResult(
                score=0.5, confidence="low", verdict="uncertain",
                details={"error": "Model not loaded"},
            )

        # Ensure correct shape
        if fakeprint.ndim == 1:
            fakeprint = fakeprint.reshape(1, -1)

        # Logistic regression: sigmoid(w . x + b)
        logit = np.dot(fakeprint, self.weights.T) + self.bias
        probability = 1.0 / (1.0 + np.exp(-logit))

        score = float(probability[0, 0])

        # Verdict with calibrated thresholds
        # Model threshold is 0.5 per config, but we add margins
        if score >= 0.85:
            verdict = "ai"
            confidence = "high"
        elif score >= 0.60:
            verdict = "ai"
            confidence = "medium"
        elif score <= 0.15:
            verdict = "human"
            confidence = "high"
        elif score <= 0.40:
            verdict = "human"
            confidence = "medium"
        else:
            verdict = "uncertain"
            confidence = "low"

        return SOTAResult(
            score=round(score, 6),
            confidence=confidence,
            verdict=verdict,
            details={
                "model": "lofcz/ai-music-detector",
                "model_type": "logistic_regression",
                "input_features": len(fakeprint),
                "logit": float(logit[0, 0]),
                "threshold": 0.5,
                "accuracy": "99.88%",
                "fpr": "0.31%",
                "fnr": "0.02%",
                "targets": "Suno <= 5, Udio <= 1.5",
            },
        )
