"""Unit tests for the corpus ingest gate — self-contained synthetic signals.

    cd tools/vox-corpus && uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
from vox_corpus import gate

SR = 44_100


def _steady_drone(dur=1.0):
    t = np.arange(int(SR * dur)) / SR
    return 0.9 * np.sin(2 * np.pi * 180.0 * t)  # gapless, flat → the metal-buzz class


def _dynamic_phrase(dur=1.5):
    """Syllable-like bursts with gaps — energy modulation a VAD keeps."""
    n = int(SR * dur)
    sig = np.zeros(n)
    t = np.arange(n) / SR
    period = int(SR * 0.35)
    for start in range(0, n, period):
        env = np.exp(-8.0 * np.arange(min(period, n - start)) / SR)
        seg = np.sin(2 * np.pi * 200.0 * t[start:start + len(env)]) * env
        sig[start:start + len(env)] += seg
    return sig * 0.9


def test_steady_drone_rejected():
    v = gate.survives(_steady_drone(), SR)
    assert v["survives"] is False
    assert any("steady-state" in r for r in v["reasons"])


def test_dynamic_phrase_admitted():
    v = gate.survives(_dynamic_phrase(), SR)
    assert v["survives"] is True
    assert v["modulation_db"] > gate.STEADY_MODULATION_DB or v["voiced_frac"] < gate.STEADY_VOICED_FRAC


def test_near_silence_rejected():
    v = gate.survives(_steady_drone() * 1e-3, SR)  # −60 dBFS
    assert v["survives"] is False
    assert any("quiet" in r for r in v["reasons"])


def test_peak_normalize():
    x = _dynamic_phrase() * 0.2
    out, in_peak = gate.peak_normalize(x, 0.9)
    assert float(np.max(np.abs(out))) == pytest.approx(0.9, abs=1e-3)
    assert in_peak == pytest.approx(0.2 * 0.9, abs=0.05)  # roughly the pre-norm peak


def test_verdict_is_json_safe():
    v = gate.survives(_dynamic_phrase(), SR)
    assert all(not isinstance(x, np.generic) for x in v.values() if not isinstance(x, list))
