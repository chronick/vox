"""Shared fixtures: a synthetic mini-dataset written to a tmp dir.

Everything is generated inline (tones + noise) so the suite runs anywhere the dataset venv is
installed — no fixture audio files in the repo. Run:

    cd tools/vox-dataset && uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

SR = 44_100


def _voiced_tone(f0_hz: float, dur: float = 6.0, n_harm: int = 15, sr: int = SR) -> np.ndarray:
    """A band-limited harmonic tone at f0 (clear periodicity so pyworld locks F0), with a short
    silent tail so an offset exists for the dryness measure."""
    t = np.arange(int(sr * dur)) / sr
    sig = sum((1.0 / k) * np.sin(2 * np.pi * f0_hz * k * t) for k in range(1, n_harm + 1))
    sig = sig / np.max(np.abs(sig)) * 0.7
    # amplitude envelope: ramp up, sustain, then hard stop 0.4 s before the end (dry offset).
    env = np.ones_like(sig)
    off = int((dur - 0.4) * sr)
    env[off:] = 0.0
    env[:int(0.02 * sr)] = np.linspace(0, 1, int(0.02 * sr))
    return (sig * env).astype("float64")


@pytest.fixture
def mini_dataset(tmp_path):
    """A 4-clip synthetic dataset:
    - a.wav  mono 44.1k, 6 s, ~110 Hz, dry, clean
    - b.wav  mono 44.1k, 6 s, ~180 Hz, dry, clean
    - clip.wav mono 44.1k, 3 s, deliberately CLIPPED
    - noisy.wav mono 44.1k, 4 s, high noise floor
    Returns the directory Path.
    """
    d = tmp_path / "mini"
    d.mkdir()

    sf.write(d / "a.wav", _voiced_tone(110.0, 6.0), SR, subtype="PCM_16")
    sf.write(d / "b.wav", _voiced_tone(180.0, 6.0), SR, subtype="PCM_16")

    # Clipped clip: amplify a tone past full scale and hard-limit.
    clipped = _voiced_tone(150.0, 3.0) * 3.0
    clipped = np.clip(clipped, -1.0, 1.0)
    sf.write(d / "clip.wav", clipped, SR, subtype="PCM_16")

    # Noisy clip: tone + strong white noise (high noise floor throughout).
    rng = np.random.default_rng(0)
    noisy = _voiced_tone(140.0, 4.0) * 0.5 + rng.normal(0, 0.15, int(SR * 4.0))
    noisy = np.clip(noisy, -0.99, 0.99)
    sf.write(d / "noisy.wav", noisy.astype("float64"), SR, subtype="PCM_16")

    return d


@pytest.fixture
def stereo_48k(tmp_path):
    """A single stereo 48 kHz 24-bit clip — for conformance checks."""
    d = tmp_path / "wrong"
    d.mkdir()
    mono = _voiced_tone(120.0, 5.0, sr=48_000)
    stereo = np.stack([mono, mono], axis=1)
    sf.write(d / "s.wav", stereo, 48_000, subtype="PCM_24")
    return d
