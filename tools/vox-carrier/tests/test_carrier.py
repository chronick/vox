"""Carrier tests — the deep-body voice.

The render tests use the **growl-55** SC body, so they gate on `say` + `ffmpeg` +
SuperCollider. They pin the two properties the carrier exists to deliver:

  1. the voice is genuinely DEEP — low-floor pyworld f0 median < 110 Hz (a naive
     65 Hz-floor tracker reads a harsh bass high off a partial), and
  2. the FLOW grid SURVIVES into the vocoded body — the wet output's amplitude
     envelope tracks the dry modulator's (lag-lenient correlation; ess_mix=0 to
     isolate the vocoded body from the additive consonant path).

    cd tools/vox-carrier && uv run pytest -q
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from vox_carrier.carrier import render_carrier_one, render_carrier_verse
from vox_flow import flow, render

PATTERN = "X.x.X.xxx.x.X..."
SYLLABLES = ["da", "grid", "tooth", "sta", "tic", "black"]
BPM = 142
SR = 44100


def _sc_available() -> bool:
    try:
        from smpl_synth.backends import sc_available
        return sc_available()
    except Exception:  # noqa: BLE001
        return False


_FULL_DEPS = (render.say_available() and render.ffmpeg_available() and _sc_available()
              and shutil.which("sclang") is not None)


def _lowfloor_f0_median(x, sr):
    """Median voiced F0 via a low-floor pyworld pass (floor=30, ceil=200) + stonemask.

    A 65 Hz-floor tracker cannot see a 55 Hz carrier and, worse, locks onto a partial and reads
    it HIGH — so a deep-carrier f0 MUST be measured low-floor or the test would silently pass a
    voice that isn't actually deep.
    """
    import pyworld as pw

    xx = np.ascontiguousarray(x, dtype="float64")
    f0, t = pw.harvest(xx, sr, f0_floor=30.0, f0_ceil=200.0, frame_period=5.0)
    f0 = pw.stonemask(xx, f0, t, sr)
    v = f0[f0 > 0]
    return float(np.median(v)) if v.size else None


def _amp_env(x, sr, win_ms=40.0, hop_ms=10.0):
    """Rectify + moving-average amplitude envelope (window >> the ~16 ms carrier period, so the
    55-62 Hz body ripple doesn't pollute it), decimated to a hop grid."""
    r = np.abs(np.ascontiguousarray(x, dtype="float64"))
    w = max(int(win_ms / 1000.0 * sr), 1)
    sm = np.convolve(r, np.ones(w) / w, mode="same")
    return sm[:: max(int(hop_ms / 1000.0 * sr), 1)]


def _corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    d = a.std() * b.std()
    return None if d < 1e-12 else float((a * b).mean() / d)


def _grid_corr_laglenient(sig, modulator, sr, max_lag_ms=80.0, hop_ms=10.0):
    """Max Pearson of the amplitude envelopes over a bounded lag search (±max_lag_ms)."""
    a = _amp_env(sig, sr, hop_ms=hop_ms)
    b = _amp_env(modulator, sr, hop_ms=hop_ms)
    ml = max(int(max_lag_ms / hop_ms), 1)
    best = -1.0
    for lag in range(-ml, ml + 1):
        aa, bb = (a[lag:], b[:len(b) - lag]) if lag > 0 else ((a[:lag], b[-lag:]) if lag < 0 else (a, b))
        c = _corr(aa, bb)
        if c is not None and c > best:
            best = c
    return best


def test_bass_chain_preserves_sub_and_ess_band():
    """The bass chain must keep the sub fundamental (highpass=55, not 180) and the consonant ess
    band (lowpass=8000, not 3400) that the grit chain deliberately deletes."""
    assert "highpass=f=55" in flow.BASS_CHAIN
    assert "lowpass=f=8000" in flow.BASS_CHAIN
    assert "highpass=f=180" not in flow.BASS_CHAIN
    assert "lowpass=f=3400" not in flow.BASS_CHAIN
    # the grid-gate / squash / slap / limiter grammar is inherited unchanged
    for stage in ("agate", "acompressor", "aecho", "alimiter"):
        assert stage in flow.BASS_CHAIN


@pytest.mark.skipif(not _FULL_DEPS, reason="say/ffmpeg/SuperCollider not all on PATH")
def test_render_carrier_one_is_deep_and_on_grid(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))

    r = render_carrier_one(PATTERN, SYLLABLES, BPM, "growl-55", sr=SR, ess_mix=0.0)

    final = r["final"]
    assert final.size > 0 and float(np.max(np.abs(final))) <= 0.95   # rendered, limited

    # (1) genuinely DEEP: low-floor f0 median well under 110 Hz.
    f0 = _lowfloor_f0_median(r["wet"], SR)
    assert f0 is not None, "no voiced frames measured"
    assert f0 < 110.0, f"the carrier must be deep (low-floor f0 < 110 Hz); got {f0:.1f} Hz"

    # (2) the FLOW grid SURVIVES into the vocoded body: envelope tracks the dry modulator.
    corr = _grid_corr_laglenient(r["wet"], r["modulator"], SR)
    assert corr > 0.5, f"FLOW grid must survive the vocode (env corr > 0.5); got {corr:.3f}"


LINES = ["Kick the pattern back to the top", "Cut the deck and count to ten"]


def _estoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Extended STOI of a degraded take against the dry modulator reference."""
    from pystoi import stoi

    n = min(len(ref), len(deg))
    return float(stoi(np.asarray(ref)[:n], np.asarray(deg)[:n], sr, extended=True))


@pytest.mark.skipif(not _FULL_DEPS, reason="say/ffmpeg/SuperCollider not all on PATH")
def test_render_carrier_verse_dry_layer_helps_and_is_deep(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))

    r = render_carrier_verse(LINES, BPM, "growl-55", sr=SR, ess_mix=0.30, dry_db=-14.0)

    final, body, mod = r["final"], r["body"], r["modulator"]
    assert final.size > 0 and float(np.max(np.abs(final))) <= 0.95   # rendered, peak-safe
    assert len(mod) > 0 and r["params"]["n_lines"] == 2

    # (1) the dry-diction layer must MEASURABLY help: ESTOI with the layer > ESTOI without it.
    e_final = _estoi(mod, final, SR)     # body + dry-diction layer
    e_body = _estoi(mod, body, SR)       # vocoded body only (the layer disabled), same render
    assert e_final > e_body, f"dry layer must raise ESTOI: final {e_final:.4f} <= body {e_body:.4f}"

    # (2) genuinely DEEP: bass-safe guarded f0 median well under 130 Hz, measured on the voiced
    # BODY (pre dry-layer): the dry-diction layer is deliberately mid-band (500-5000 Hz), so
    # measuring the blend would read the diction, not the voice.
    from vox_core import measure_f0_guarded
    f0 = measure_f0_guarded(body, SR, target_hint=55.0)["f0_hz"]
    assert f0 is not None, "no voiced frames measured"
    assert f0 < 130.0, f"the verse body must stay deep (guarded f0 < 130 Hz); got {f0:.1f} Hz"
