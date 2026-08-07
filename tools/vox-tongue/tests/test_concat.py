"""Tests for the concat backend — stitching spliced syllabank clips into a sung line.

A tiny hand-built bank (synthesised harmonic tones, license-clean ``own-render``) in a tmp root
keeps the test hermetic: no dependence on a user bank. Covers the manifest/provenance contract,
per-syllable WORLD pitch imposition (low-floor measured), the say fallback for null phones, the
duration-fill (short clip -> long note -> stretched, ratio recorded), and compile_packet's
refusal of a rewrite-verdict line.

Run (siblings on PYTHONPATH):

    cd tools/vox-tongue && uv run pytest tests/test_concat.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from vox_tongue import compile as compile_mod
from vox_tongue import concat as concat_mod
from vox_tongue.render import say_available

SR = 44_100


def _tone(f0: float, dur_s: float, sr: int = SR) -> np.ndarray:
    """A voiced harmonic tone at ``f0`` (6 partials + short fades) — WORLD-trackable source."""
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    x = sum((1.0 / k) * np.sin(2.0 * np.pi * k * f0 * t) for k in range(1, 7))
    fade = int(0.01 * sr)
    if fade > 1 and n > 2 * fade:
        x[:fade] *= np.linspace(0.0, 1.0, fade)
        x[-fade:] *= np.linspace(1.0, 0.0, fade)
    peak = float(np.max(np.abs(x))) or 1.0
    return (x / peak * 0.9).astype("float64")


def _low_floor_median(seg: np.ndarray, sr: int = SR):
    """Median voiced F0 via a LOW-FLOOR pyworld pass (30 Hz) — the bass-safe measurement."""
    import pyworld as pw

    x = np.ascontiguousarray(seg.astype("float64"))
    f0, t = pw.harvest(x, sr, f0_floor=30.0, f0_ceil=400.0, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    voiced = f0[f0 > 0]
    return float(np.median(voiced)) if voiced.size else None


def _build_bank(root, entries):
    """Add ``entries`` = list of (id, phones, tone_f0, dur_s) as own-render bank rows in ``root``."""
    from vox_syllabank import bank as B

    for eid, phones, f0, dur in entries:
        B.add_entry(str(root), _tone(f0, dur), SR, {
            "id": eid, "syllable": eid, "phones": phones, "f0_hz": f0,
            "source": "own-render:test-tone", "license": "own-render", "attribution": None,
        })


# ---------------------------------------------------------------------------
# manifest + provenance + per-syllable pitch.
# ---------------------------------------------------------------------------
def test_concat_manifest_and_pitch(tmp_path):
    root = tmp_path / "bank"
    _build_bank(root, [
        ("tone-mah", ["M", "AH0"], 150.0, 0.4),
        ("tone-shiyn", ["SH", "IY1", "N"], 180.0, 0.4),
        ("tone-seh", ["S", "EH1"], 160.0, 0.4),
    ])

    bpm = 100.0
    # 2-syllable score, both with phones present -> both resolve from the bank.
    score = compile_mod.compile(["machine"], ["A2", "E2"], bpm=bpm)
    assert len(score["syllables"]) == 2

    samples, manifest = concat_mod.render_concat(score, bank_root=str(root), sr=SR)
    assert samples.size > 0
    rows = manifest["syllables"]
    assert len(rows) == 2

    # Every row carries a license + tier + a bank id (no say fallback here).
    for r in rows:
        assert r["license"], f"row {r['text']!r} has no license"
        assert r["tier"] is not None, f"row {r['text']!r} has no tier"
        assert r["bank_id"] != "say"
        assert "time_ratio" in r
    assert manifest["provenance"]["unlicensed"] == 0
    assert manifest["provenance"]["n_say_fallback"] == 0

    # Per-syllable WORLD pitch imposition: measured (low-floor) median within 8% of target.
    from vox_larynx import world

    for r in rows:
        target = world.note_to_hz(r_text_note(score, r["text"]))
        start, end = int(r["t"] * SR), int((r["t"] + r["dur"]) * SR)
        med = _low_floor_median(samples[start:end].astype("float64"))
        assert med is not None, f"{r['text']!r} came out unvoiced"
        rel = abs(med - target) / target
        assert rel < 0.08, f"{r['text']!r}: {med:.1f} Hz vs {target:.1f} Hz ({rel:.2%})"


def r_text_note(score, text):
    for s in score["syllables"]:
        if s["text"] == text:
            return s["note"]
    raise KeyError(text)


# ---------------------------------------------------------------------------
# say fallback for null phones.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not say_available(), reason="macOS `say` not available")
def test_concat_missing_phones_say_fallback(tmp_path):
    root = tmp_path / "bank"
    _build_bank(root, [("tone-mah", ["M", "AH0"], 150.0, 0.4)])

    # A hand-built score with a null-phones syllable -> must fall back to `say`.
    score = {
        "meta": {"bpm": 100.0, "key": None, "title": None},
        "syllables": [
            {"text": "ma", "word": "machine", "phones": ["M", "AH0"], "start_beat": 0.0,
             "dur_beats": 1.0, "note": "A2", "dyn": 1.0, "articulation": None},
            {"text": "zzz", "word": "zzz", "phones": None, "start_beat": 1.0,
             "dur_beats": 1.0, "note": "E2", "dyn": 1.0, "articulation": None},
        ],
    }
    samples, manifest = concat_mod.render_concat(score, bank_root=str(root), sr=SR)
    rows = manifest["syllables"]
    assert rows[0]["bank_id"] == "tone-mah"
    assert rows[1]["bank_id"] == "say", "null-phones syllable should fall back to say"
    assert rows[1]["license"] == "own-render"
    assert manifest["provenance"]["n_say_fallback"] == 1
    assert manifest["provenance"]["unlicensed"] == 0


# ---------------------------------------------------------------------------
# duration-fill: a short clip under a long note gets time-stretched.
# ---------------------------------------------------------------------------
def test_concat_duration_fill(tmp_path):
    root = tmp_path / "bank"
    _build_bank(root, [("tone-short", ["M", "AH0"], 150.0, 0.1)])  # 0.1s clip

    # bpm 60 => 1 beat = 1.0s; a 0.1s clip must stretch to fill (ratio capped at 4.0).
    score = {
        "meta": {"bpm": 60.0, "key": None, "title": None},
        "syllables": [
            {"text": "ma", "word": "machine", "phones": ["M", "AH0"], "start_beat": 0.0,
             "dur_beats": 1.0, "note": "A2", "dyn": 1.0, "articulation": None},
        ],
    }
    samples, manifest = concat_mod.render_concat(score, bank_root=str(root), sr=SR)
    row = manifest["syllables"][0]
    assert row["time_ratio"] > 1.0, "short clip under a long note should be stretched"
    assert row["time_ratio"] == pytest.approx(4.0), "0.1s clip / 1.0s note should hit the 4x cap"


# ---------------------------------------------------------------------------
# compile_packet refuses a rewrite-verdict line (the gates finally bite).
# ---------------------------------------------------------------------------
def test_compile_packet_refuses_rewrite():
    from vox_lyric.packet import build_packet

    # A blocklisted line -> verdict "rewrite".
    packet = build_packet(["darkness eternal soul"], "percussive")
    assert packet["lines"][0]["verdict"] == "rewrite"
    with pytest.raises(ValueError, match="rewrite"):
        compile_mod.compile_packet(packet, ["E2", "G2"], bpm=142)


def test_compile_packet_keeps_and_places_percussive():
    """A kept percussive line compiles from the packet with flow_hint-derived placement."""
    from vox_lyric.packet import build_packet

    packet = build_packet(["spit the code back, kick the deck"], "percussive")
    assert packet["lines"][0]["verdict"] == "keep"
    score = compile_mod.compile_packet(packet, ["E2", "G2"], bpm=142)
    # One syllable per packet syllable; phones taken from the packet, not re-derived.
    assert len(score["syllables"]) == packet["lines"][0]["syllable_count"]
    assert score["syllables"][0]["phones"] == ["S", "P", "IH1", "T"]
    assert score["syllables"][0]["articulation"] == "spat"
