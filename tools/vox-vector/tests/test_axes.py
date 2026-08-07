"""Unit tests for axis mapping + diff — self-contained synthetic signals.

    cd tools/vox-vector && uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
from vox_vector import axes

SR = 44_100


def _tone(f0=180.0, dur=0.8):
    t = np.arange(int(SR * dur)) / SR
    sig = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 20))
    return sig / np.max(np.abs(sig)) * 0.9


def test_measure_returns_all_axes():
    r = axes.measure(_tone(), SR)
    assert set(r["vector"]) == set(axes.AXES)
    # heavy-tier axes are honestly absent in v0
    assert r["vector"]["intelligibility"] is None and r["vector"]["multiplicity"] is None


def test_harmonic_tone_low_breathiness_low_roughness():
    v = axes.measure(_tone(), SR)["vector"]
    assert v["breathiness"] < 0.3    # a clean tone is not aspirate
    assert v["roughness"] < 0.2      # …and harmonic, not clangorous


def test_noise_high_breathiness():
    rng = np.random.default_rng(0)
    v = axes.measure(rng.standard_normal(int(SR * 0.8)) * 0.3, SR)["vector"]
    assert v["breathiness"] > 0.4    # broadband noise → aspirate axis high


def test_dry_sustained_tone_is_not_spatial():
    # a tone held to the very end has no offset → spatiality reads ~0 (honest), not falsely vast
    v = axes.measure(_tone(dur=0.8), SR)["vector"]
    assert v["spatiality"] < 0.2


def test_decay_tail_reads_more_spatial_than_gated():
    base = _tone(dur=0.4)
    n = len(base)
    gated = np.concatenate([base, np.zeros(n)])                     # stops dead → dry
    reverbed = np.concatenate([base, base[::-1] * np.linspace(0.5, 0.0, n)])  # lingering tail
    assert (axes.measure(reverbed, SR)["vector"]["spatiality"]
            > axes.measure(gated, SR)["vector"]["spatiality"])


def test_diff_math():
    programmed = {"breathiness": 0.2, "roughness": 0.1, "intelligibility": 0.9}
    measured = {"breathiness": 0.25, "roughness": 0.1, "intelligibility": None, "spatiality": 0.3}
    d = axes.diff(programmed, measured)
    assert d["per_axis_error"]["breathiness"] == pytest.approx(0.05)
    assert d["per_axis_error"]["roughness"] == pytest.approx(0.0)
    assert "intelligibility" in d["skipped"]     # None on one side → skipped
    assert d["mean_abs_error"] == pytest.approx(0.025)


def test_upstream_descriptors_used():
    # supplying an ear descriptor overrides the internal estimate (breathiness from flatness_hf)
    v = axes.measure(_tone(), SR, upstream={"voice.spectral_flatness_hf": 0.15,
                                            "voice.inharmonicity": 0.8,
                                            "voice.hnr_db": 18.0, "voice.jitter_local": 0.01})["vector"]
    assert v["roughness"] == pytest.approx(0.8, abs=0.01)
    assert v["humanness"] is not None
