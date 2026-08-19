"""Remastering Pipeline Processors.

Contains Demucs separation, humanizer, spectral processing, spatial tools, and master chain.
Lazy imports for heavy dependencies (torch/audio_separator) to allow fast startup.
"""

from .humanizer import Humanizer
from .spectral import SpectralProcessor
from .phase import PhasePerturbator
from .spatial import SpatialProcessor
from .noise import NoiseProcessor
from .master import MasterChain, ReferenceMaster
from .stft_perturbator import SpectralPerturbator

# Lazy-load stem separator (requires torch) — call get_stem_separator() to access
_stem_separator_cls = None

def get_stem_separator():
    global _stem_separator_cls
    if _stem_separator_cls is None:
        from .stem_separator import StemSeparator
        _stem_separator_cls = StemSeparator
    return _stem_separator_cls


__all__ = [
    "StemSeparator",
    "get_stem_separator",
    "Humanizer",
    "SpectralProcessor",
    "PhasePerturbator",
    "SpatialProcessor",
    "NoiseProcessor",
    "MasterChain",
    "ReferenceMaster",
]
