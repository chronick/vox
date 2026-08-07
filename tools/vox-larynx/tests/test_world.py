"""Unit tests for the WORLD DSP core — deterministic, self-contained (no fixture files, no CLI).

Each test synthesises a periodic voiced signal inline (sum of harmonics = a clear F0 for WORLD
to track), so the suite runs anywhere the larynx venv is installed. Run:

    cd tools/vox-larynx && uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
from vox_larynx import world

SR = 44_100


def _voiced(f0_hz: float, dur: float = 0.8, n_harm: int = 20) -> np.ndarray:
    """A band-limited pulse-ish tone at ``f0_hz`` (clear periodicity → WORLD locks F0)."""
    t = np.arange(int(SR * dur)) / SR
    sig = sum((1.0 / k) * np.sin(2 * np.pi * f0_hz * k * t) for k in range(1, n_harm + 1))
    return (sig / np.max(np.abs(sig)) * 0.9).astype("float64")


def test_note_to_hz():
    assert world.note_to_hz("A4") == pytest.approx(440.0)
    assert world.note_to_hz("A3") == pytest.approx(220.0)
    assert world.note_to_hz("C4") == pytest.approx(261.63, abs=0.1)
    assert world.note_to_hz("C#4") == pytest.approx(277.18, abs=0.1)


def test_analyze_tracks_f0():
    f0, sp, ap, t = world.analyze(_voiced(180.0), SR)
    stats = world.f0_stats(f0)
    assert stats["voiced_frac"] > 0.8
    assert stats["f0_median_hz"] == pytest.approx(180.0, abs=3.0)
    # all values native python (JSON-safe frame data)
    assert all(not isinstance(v, np.generic) for v in stats.values())


def test_render_transpose_octave():
    y, params = world.render(_voiced(180.0), SR, semitones=12.0)
    f0, *_ = world.analyze(y, SR)
    assert world.f0_stats(f0)["f0_median_hz"] == pytest.approx(360.0, rel=0.04)


def test_render_to_hz_monotone():
    y, params = world.render(_voiced(180.0), SR, to_hz=200.0)
    f0, *_ = world.analyze(y, SR)
    v = f0[f0 > 0]
    assert np.median(v) == pytest.approx(200.0, abs=4.0)
    assert np.std(v) < 6.0  # monotone → tight distribution


def test_render_flatten():
    # glide input, flatten → near-constant output pitch
    n = int(SR * 0.8)
    f0_curve = np.geomspace(150.0, 300.0, n)
    t = np.arange(n) / SR
    phase = np.cumsum(f0_curve / SR)
    sig = sum((1.0 / k) * np.sin(2 * np.pi * k * phase) for k in range(1, 20))
    x = (sig / np.max(np.abs(sig)) * 0.9)
    y, _ = world.render(x, SR, flatten=True)
    f0, *_ = world.analyze(y, SR)
    v = f0[f0 > 0]
    assert np.std(v) < np.std(f0_curve) * 0.5  # much flatter than the input glide


def test_harmonize_stack():
    y, params = world.harmonize(_voiced(220.0), SR, chord=(0, 4, 7), drone=True)
    assert params["voices"] == 5  # lead + 3 chord + drone
    assert np.max(np.abs(y)) <= 0.95  # headroom preserved (no clip)
    assert len(y) >= int(SR * 0.5)


def test_harmonize_deterministic():
    a, _ = world.harmonize(_voiced(220.0), SR, chord=(0, 3, 7), seed=1)
    b, _ = world.harmonize(_voiced(220.0), SR, chord=(0, 3, 7), seed=1)
    assert np.allclose(a, b)  # fixed seed → reproducible (memoizable)
