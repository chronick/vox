"""Unit tests for voice descriptors — self-contained synthetic signals (no fixture files/CLI).

    cd tools/vox-ear && uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
from vox_ear import descriptors

SR = 44_100


def _formant_vowel(f0_hz, formants, dur=0.9, vib_rate=0.0, vib_cents=0.0, jitter=0.004, seed=0):
    """Source-filter /vowel/: a jittered impulse-train glottal source through 2-pole formant
    resonators. The small jitter is what makes parselmouth's autocorrelation tracker read it as
    voiced (a dead-steady synthetic tone reads as unvoiced)."""
    from scipy.signal import lfilter
    n = int(SR * dur)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    f0 = f0_hz * (2.0 ** ((vib_cents / 1200.0) * np.sin(2 * np.pi * vib_rate * t)) if vib_rate else np.ones(n))
    src = np.zeros(n)
    phase = 0.0
    for i in range(n):
        phase += f0[i] * (1.0 + jitter * rng.standard_normal()) / SR
        if phase >= 1.0:
            phase -= 1.0
            src[i] = 1.0
    out = np.zeros(n)
    for fc, bw in formants:
        r = np.exp(-np.pi * bw / SR); th = 2 * np.pi * fc / SR
        out += lfilter([1 - r], [1.0, -2 * r * np.cos(th), r * r], src)
    return out / (np.max(np.abs(out)) or 1.0) * 0.9


A_FORMANTS = [(730.0, 80.0), (1090.0, 90.0), (2440.0, 120.0)]
I_FORMANTS = [(270.0, 60.0), (2290.0, 100.0), (3010.0, 130.0)]


def test_vowel_a_descriptors():
    d = descriptors.describe(_formant_vowel(180.0, A_FORMANTS, vib_rate=5.0, vib_cents=30.0), SR)
    assert d["voice.f0_median_hz"] == pytest.approx(180.0, abs=4.0)
    assert 550 < d["voice.f1_hz"] < 950      # /a/ has a high F1
    assert d["voice.hnr_db"] > 15            # clean voiced
    assert d["voice.voiced_frac"] > 0.8
    assert all(not isinstance(v, np.generic) for v in d.values())  # JSON-safe


def test_vowel_i_has_low_f1_high_f2():
    d = descriptors.describe(_formant_vowel(200.0, I_FORMANTS), SR)
    assert d["voice.f1_hz"] < 450            # /i/ has a low F1
    assert d["voice.f2_hz"] > 1800           # …and a high F2
    assert d["voice.f2_hz"] > d["voice.f1_hz"]


def test_breath_is_unvoiced():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(int(SR * 0.8))
    d = descriptors.describe(noise * 0.3, SR)
    assert d["voice.voiced_frac"] < 0.3
    assert d["voice.f0_median_hz"] is None    # no periodicity → timbre keys null
    assert d["voice.spectral_flatness_hf"] > 0.1  # noise is spectrally flat


def test_vibrato_detected():
    d = descriptors.describe(_formant_vowel(220.0, A_FORMANTS, vib_rate=5.5, vib_cents=40.0), SR)
    assert d["voice.vibrato_rate_hz"] is not None
    assert 4.0 < d["voice.vibrato_rate_hz"] < 7.0   # recovers the 5.5 Hz vibrato


def test_hnr_orders_clean_above_breathy():
    clean = descriptors.describe(_formant_vowel(180.0, A_FORMANTS), SR)
    rng = np.random.default_rng(1)
    # breathy but still voiced: a modest noise floor lowers HNR without killing pitch detection
    breathy_sig = _formant_vowel(180.0, A_FORMANTS) + 0.12 * rng.standard_normal(int(SR * 0.9))
    breathy = descriptors.describe(breathy_sig, SR)
    assert breathy["voice.hnr_db"] is not None
    assert clean["voice.hnr_db"] > breathy["voice.hnr_db"]  # noise lowers HNR
