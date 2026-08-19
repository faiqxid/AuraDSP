import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import soundfile as sf
import tempfile

from processors.master import ReferenceMaster

@pytest.fixture
def dummy_audio():
    # 1 second of 440Hz sine wave (stereo)
    sr = 44100
    t = np.linspace(0, 1.0, sr)
    y = np.stack([np.sin(2*np.pi*440*t)*0.5, np.sin(2*np.pi*440*t)*0.5], axis=-1)
    return y, sr

@pytest.fixture
def reference_file():
    # 1 second of 880Hz sine wave, slightly louder
    sr = 44100
    t = np.linspace(0, 1.0, sr)
    y = np.stack([np.sin(2*np.pi*880*t)*0.8, np.sin(2*np.pi*880*t)*0.8], axis=-1)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
        yield f.name
    Path(f.name).unlink()

def test_reference_master_initialization():
    rm = ReferenceMaster(sr=44100)
    assert rm.sr == 44100

def test_reference_master_profile_extraction(reference_file):
    rm = ReferenceMaster(sr=44100)
    profile = rm.extract_profile(reference_file)
    
    # Check if expected keys are present
    assert "lufs" in profile
    assert "peak" in profile
    assert "rms" in profile
    assert "dynamic_range_db" in profile
    assert "spectral_centroid" in profile
    assert profile["dynamic_range_db"] > 0
    
    # 880Hz sine wave should have centroid near 880
    assert 800 < profile["spectral_centroid"] < 950

def test_reference_master_processing(dummy_audio, reference_file):
    y, sr = dummy_audio
    rm = ReferenceMaster(sr=sr)
    
    # Process audio matching the reference file
    out = rm.process(y, reference_path=reference_file)
    
    # Shape shouldn't change
    assert out.shape == y.shape
    
    # Should not be identically equal to input
    assert not np.allclose(y, out)

def test_reference_master_dynamic_matching():
    """_match_dynamics compresses audio with high peak-to-RMS toward target DR."""
    sr = 44100
    rm = ReferenceMaster(sr=sr)

    # Target: 2 second audio, 0.9 amplitude, then silence with noise
    y = np.zeros((sr*2, 2))
    y[:int(sr*1), :] = np.sin(2*np.pi*440*np.linspace(0, 1.0, sr))[:, None]
    y[:int(sr*1), :] *= 0.9  # loud 1 second
    y += np.random.randn(*y.shape) * 0.001  # tiny noise throughout
    
    # Compute DR before
    peak_in = float(np.max(np.abs(y)))
    rms_in = float(np.sqrt(np.mean(y**2)))
    dr_in = 20 * np.log10(peak_in / max(rms_in, 1e-9))
    
    # Apply compression toward a very low target DR (3dB — like a flat sine wave)
    target_dr = 3.0
    out = rm._match_dynamics(y, target_dr_db=target_dr)
    
    peak_out = float(np.max(np.abs(out)))
    rms_out = float(np.sqrt(np.mean(out**2)))
    dr_out = 20 * np.log10(peak_out / max(rms_out, 1e-9))
    
    # dr_in should be significantly wider than target
    assert dr_in > target_dr + 3.0, f"dr_in={dr_in:.2f} should be > {target_dr + 3.0}"
    # Compressor should reduce DR (not increase it)
    assert dr_out < dr_in, f"Compression should reduce DR: dr_out={dr_out:.2f}, dr_in={dr_in:.2f}"
    # Peak should be same or lower after compression
    assert peak_out <= peak_in + 1e-6, f"Peak should not increase: peak_out={peak_out:.4f}, peak_in={peak_in:.4f}"
