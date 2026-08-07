"""Tests for the singing-warp op (warp.py) — the alignment + WORLD-warp that puts syllable onsets
on the grid and stretches held notes to note length.

The whisper-dependent path (``align_words`` / ``warp_to_score``) is gated behind
``whisper_available`` (faster-whisper is a heavy optional dep). The pure-DSP pieces — syllable
anchoring, the ratio clamp, and join continuity — are tested with hand-built synthetic audio so
they run everywhere.

Run (siblings on PYTHONPATH):

    cd tools/vox-tongue && uv run pytest tests/test_warp.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from vox_tongue import compile as compile_mod
from vox_tongue import warp as warp_mod

SR = 44_100


def _tone(f0: float, dur_s: float, sr: int = SR, gap_s: float = 0.0) -> np.ndarray:
    """A voiced harmonic tone (6 partials + fades), optional trailing silence — WORLD-trackable."""
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    x = sum((1.0 / k) * np.sin(2.0 * np.pi * k * f0 * t) for k in range(1, 7))
    fade = int(0.01 * sr)
    if fade > 1 and n > 2 * fade:
        x[:fade] *= np.linspace(0.0, 1.0, fade)
        x[-fade:] *= np.linspace(1.0, 0.0, fade)
    x = (x / (np.max(np.abs(x)) or 1.0) * 0.9)
    if gap_s > 0:
        x = np.concatenate([x, np.zeros(int(sr * gap_s))])
    return x.astype("float64")


# ---------------------------------------------------------------------------
# syllable_anchors — proportional phone-count split (no whisper needed).
# ---------------------------------------------------------------------------
def test_syllable_anchors_split_by_phone_count():
    # "machine" -> 2 syllables, phones ["M","AH0"] (2) + ["SH","IY1","N"] (3): a 2:3 split.
    score = compile_mod.compile(["machine"], ["A3"], bpm=120)
    assert len(score["syllables"]) == 2
    words = [{"word": "machine", "t0": 0.0, "t1": 1.0, "matched": True}]
    anchors = warp_mod.syllable_anchors(words, score)
    assert len(anchors) == 2
    # First syllable ~2/5 of the span, second ~3/5.
    assert anchors[0]["start"] == pytest.approx(0.0, abs=1e-6)
    assert anchors[0]["end"] == pytest.approx(0.4, abs=1e-3)
    assert anchors[1]["start"] == pytest.approx(0.4, abs=1e-3)
    assert anchors[1]["end"] == pytest.approx(1.0, abs=1e-6)


def test_syllable_anchors_multiword_grouping():
    score = compile_mod.compile(["I lay low"], ["A3", "C4", "E4"], bpm=120)
    words = [
        {"word": "i", "t0": 0.0, "t1": 0.5, "matched": True},
        {"word": "lay", "t0": 0.5, "t1": 1.2, "matched": True},
        {"word": "low", "t0": 1.2, "t1": 2.0, "matched": True},
    ]
    anchors = warp_mod.syllable_anchors(words, score)
    assert [a["word"] for a in anchors] == ["I", "lay", "low"]
    assert anchors[1]["start"] == pytest.approx(0.5, abs=1e-6)
    assert anchors[2]["end"] == pytest.approx(2.0, abs=1e-6)


def test_syllable_anchors_lowconf_span_clamped_to_neighbours():
    """A word whisper never matched (matched=False) whose interpolated span is a long outlier is
    capped to LOWCONF_SYL_DUR_CAP x the median per-syllable span of the confident words."""
    score = compile_mod.compile(["I lay low"], ["A3", "C4", "E4"], bpm=120)
    # "i" and "low" are confident single-syllable words at 0.5 s each -> median per-syllable = 0.5 s.
    # "lay" was interpolated to a 1.0 s span (0.5..1.5) -> capped to 1.5 * 0.5 = 0.75 s.
    words = [
        {"word": "i", "t0": 0.0, "t1": 0.5, "matched": True},
        {"word": "lay", "t0": 0.5, "t1": 1.5, "matched": False},
        {"word": "low", "t0": 1.5, "t1": 2.0, "matched": True},
    ]
    anchors = warp_mod.syllable_anchors(words, score)
    assert anchors[1]["start"] == pytest.approx(0.5, abs=1e-6)
    assert anchors[1]["end"] == pytest.approx(1.25, abs=1e-3)  # clamped, not the raw 1.5


def test_syllable_anchors_lowconf_within_cap_unchanged():
    """A low-confidence word already within the neighbour cap keeps its full span (no shrink)."""
    score = compile_mod.compile(["I lay low"], ["A3", "C4", "E4"], bpm=120)
    words = [
        {"word": "i", "t0": 0.0, "t1": 0.5, "matched": True},
        {"word": "lay", "t0": 0.5, "t1": 1.1, "matched": False},  # 0.6 s < 0.75 s cap
        {"word": "low", "t0": 1.5, "t1": 2.0, "matched": True},
    ]
    anchors = warp_mod.syllable_anchors(words, score)
    assert anchors[1]["end"] == pytest.approx(1.1, abs=1e-6)


# ---------------------------------------------------------------------------
# _warp_segment — WORLD elastic time-scale + the ratio clamp.
# ---------------------------------------------------------------------------
def test_warp_segment_ratio_clamped_high():
    from vox_larynx import world

    seg = _tone(180.0, 0.20)  # 0.2 s source, ask for a 2.0 s note -> raw ratio 10, clamp to 4.
    warped, ratio = warp_mod._warp_segment(seg, SR, 2.0, world.note_to_hz("A3"), world)
    assert ratio == pytest.approx(warp_mod.TIME_RATIO_MAX)
    # Achieved dur is actual*clamped_ratio (~0.8 s), NOT the impossible 2.0 s — honest clamp.
    assert len(warped) / SR == pytest.approx(0.8, abs=0.05)


def test_warp_segment_ratio_clamped_low():
    from vox_larynx import world

    seg = _tone(180.0, 2.0)  # 2 s source squeezed toward 0.1 s -> raw ratio 0.05, clamp to 0.25.
    warped, ratio = warp_mod._warp_segment(seg, SR, 0.1, world.note_to_hz("A3"), world)
    assert ratio == pytest.approx(warp_mod.TIME_RATIO_MIN)


def test_warp_segment_imposes_pitch():
    from vox_larynx import world

    seg = _tone(180.0, 0.6)
    warped, _ratio = warp_mod._warp_segment(seg, SR, 0.6, world.note_to_hz("A3"), world)  # A3=220
    import pyworld as pw

    x = np.ascontiguousarray(warped)
    f0, t = pw.harvest(x, SR, f0_floor=80.0, f0_ceil=500.0, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, SR)
    med = float(np.median(f0[f0 > 0]))
    assert med == pytest.approx(220.0, rel=0.06)  # re-voiced to A3, not the 180 Hz source


# ---------------------------------------------------------------------------
# Join continuity — the equal-power crossfade must not leave a step at a seam.
# ---------------------------------------------------------------------------
def test_join_continuity_no_step():
    # Two abutting grid slots; assembling must not produce a >6 dB inter-sample jump at the seam.
    score = compile_mod.compile(["lay low"], ["A3", "C4"], bpm=120)  # beats 0 and 1, 1 s each
    # Build a synthetic "vocal": two 0.4 s tones with silence, whose word spans we pin directly.
    words = [
        {"word": "lay", "t0": 0.0, "t1": 0.45, "matched": True},
        {"word": "low", "t0": 0.5, "t1": 0.95, "matched": True},
    ]
    _ = words  # spans below are pinned via the mocked aligner so the SEAM is the shipped crossfade
    vocal = np.concatenate([_tone(200.0, 0.45), np.zeros(int(0.05 * SR)), _tone(160.0, 0.45)])

    def fake_align(v, sr, expected_text, **kw):
        return ([{"word": "lay", "t0": 0.0, "t1": 0.45, "matched": True},
                 {"word": "low", "t0": 0.5, "t1": 0.95, "matched": True}],
                {"n_expected": 2, "n_recognised": 2, "n_matched": 2,
                 "mismatches": [], "transcript": "lay low"})

    from unittest import mock
    with mock.patch.object(warp_mod, "align_words", fake_align):
        out, report = warp_mod.warp_to_score(vocal, SR, score, bpm=120)

    # The interior seam is the 2nd syllable's grid onset (beat 1 = 0.5 s). Check the smoothed
    # amplitude envelope across a window straddling it — no >6 dB step (a factor of 2.0) means the
    # equal-power crossfade held (no notch/click). The global attack/release are excluded on
    # purpose: a note starting from silence is a legitimate >6 dB rise, not a seam click.
    seam_s = report["syllables"][1]["target_t"]
    win = int(0.005 * SR)
    rms = np.sqrt(np.convolve(out.astype("float64") ** 2, np.ones(win) / win, mode="same") + 1e-12)
    lo, hi = int((seam_s - 0.03) * SR), int((seam_s + 0.03) * SR)
    frames = rms[lo:hi:win // 2]
    step_db = 20.0 * np.log10(np.maximum(frames[1:], frames[:-1]) /
                              np.minimum(frames[1:], frames[:-1]))
    assert float(np.max(step_db)) < 6.0


# ---------------------------------------------------------------------------
# align_words + full warp_to_score — whisper-gated end-to-end.
# ---------------------------------------------------------------------------
whisper_gate = pytest.mark.skipif(not warp_mod.whisper_available(),
                                  reason="faster-whisper not installed")


def _clicky_word_audio():
    """Synthesise 'lay low' as two clearly separated voiced tones — whisper won't transcribe tones
    reliably, so this fixture is only for the syllable-anchor/warp math with a MOCKED aligner."""
    return np.concatenate([_tone(200.0, 0.4, gap_s=0.2), _tone(160.0, 0.4, gap_s=0.2)])


def test_warp_to_score_places_on_grid_with_mocked_alignment(monkeypatch):
    """Grid placement correctness WITHOUT whisper: mock align_words to return known word spans and
    assert every syllable onset lands within 30 ms of its grid slot (the acceptance metric)."""
    score = compile_mod.compile(["lay low"], ["A3", "C4"], bpm=90)
    vocal = _clicky_word_audio()  # word1 ~[0,0.4], word2 ~[0.6,1.0]

    def fake_align(v, sr, expected_text, **kw):
        words = [
            {"word": "lay", "t0": 0.0, "t1": 0.4, "matched": True},
            {"word": "low", "t0": 0.6, "t1": 1.0, "matched": True},
        ]
        return words, {"n_expected": 2, "n_recognised": 2, "n_matched": 2,
                       "mismatches": [], "transcript": "lay low"}

    monkeypatch.setattr(warp_mod, "align_words", fake_align)
    out, report = warp_mod.warp_to_score(vocal, SR, score, bpm=90)
    spb = 60.0 / 90.0
    for i, row in enumerate(report["syllables"]):
        target = i * spb
        assert row["target_t"] == pytest.approx(target, abs=1e-3)
        assert abs(row["onset_err_ms"]) <= 30.0  # onset on the grid within 30 ms


@whisper_gate
def test_align_words_matches_expected_sequence():
    """faster-whisper on a real spoken clip — say-render 'lay low' and confirm the aligner returns
    two ordered word spans matched to the expected sequence."""
    from vox_tongue.render import say_available

    if not say_available():
        pytest.skip("`say` unavailable")
    from vox_tongue.render import _say_render

    clip = _say_render("lay low", "Fred", SR)
    words, report = warp_mod.align_words(clip, SR, "lay low")
    assert [w["word"] for w in words] == ["lay", "low"]
    assert words[0]["t1"] <= words[1]["t0"] + 0.05  # ordered, non-overlapping-ish
    assert report["n_expected"] == 2


# ---------------------------------------------------------------------------
# Warp MODES — the diction↔grid aggression dial (beads I2).
# ---------------------------------------------------------------------------
def test_warp_mode_unknown_raises():
    """An unknown mode fails loudly (before any whisper pass) rather than silently doing 'full'."""
    score = compile_mod.compile(["lay low"], ["A3", "C4"], bpm=120)
    with pytest.raises(ValueError):
        warp_mod.warp_to_score(_tone(180.0, 0.4), SR, score, bpm=120, mode="bogus")


def test_warp_mode_pitch_imposition_dispatch(monkeypatch):
    """Only 'full' imposes the score note per syllable; 'time-only'/'onsets'/'median-pitch' keep the
    take's native contour (pitch_imposed False) and echo their mode in the report."""
    score = compile_mod.compile(["lay low"], ["A3", "C4"], bpm=90)
    vocal = _clicky_word_audio()

    def fake_align(v, sr, expected_text, **kw):
        return ([{"word": "lay", "t0": 0.0, "t1": 0.4, "matched": True},
                 {"word": "low", "t0": 0.6, "t1": 1.0, "matched": True}],
                {"n_expected": 2, "n_recognised": 2, "n_matched": 2,
                 "mismatches": [], "transcript": "lay low"})

    monkeypatch.setattr(warp_mod, "align_words", fake_align)

    _out, rep_full = warp_mod.warp_to_score(vocal, SR, score, bpm=90, mode="full")
    assert rep_full["mode"] == "full"
    assert all(r["pitch_imposed"] for r in rep_full["syllables"])

    for m in ("time-only", "onsets", "median-pitch"):
        _out, rep = warp_mod.warp_to_score(vocal, SR, score, bpm=90, mode=m)
        assert rep["mode"] == m
        assert not any(r["pitch_imposed"] for r in rep["syllables"])
    # median-pitch attaches the global-shift record (contour-preserving register correction).
    _out, rep_mp = warp_mod.warp_to_score(vocal, SR, score, bpm=90, mode="median-pitch")
    assert set(rep_mp["global_shift"]) == {"take_median_hz", "score_median_hz", "shift_semitones"}


def test_warp_mode_onsets_ratio_capped():
    """'onsets'/'median-pitch' clamp the time-ratio to ONSET_RATIO_CAP — the interior is only nudged,
    never note-filled (the onset lands on grid by placement, not by a 4x stretch)."""
    from vox_larynx import world

    cap = warp_mod.ONSET_RATIO_CAP
    # 0.2 s source asked to fill a 2.0 s note: full mode would stretch ~10x (clamp 4x); onsets caps 1.5x.
    seg = _tone(180.0, 0.2)
    _warped, ratio = warp_mod._warp_segment(seg, SR, 2.0, None, world,
                                            ratio_min=1.0 / cap, ratio_max=cap)
    assert ratio == pytest.approx(cap)
    assert len(_warped) / SR == pytest.approx(0.2 * cap, abs=0.03)  # only the capped stretch applied
    # And the compress side clamps to 1/cap.
    seg2 = _tone(180.0, 1.0)
    _w2, ratio2 = warp_mod._warp_segment(seg2, SR, 0.1, None, world,
                                         ratio_min=1.0 / cap, ratio_max=cap)
    assert ratio2 == pytest.approx(1.0 / cap)


def test_median_shift_semitones_math():
    """The global register shift is 12*log2(target/take) — a constant interval that preserves the
    take's contour. Octaves are exact; missing/non-positive inputs mean no shift."""
    assert warp_mod._median_shift_semitones(110.0, 220.0) == pytest.approx(12.0)
    assert warp_mod._median_shift_semitones(220.0, 110.0) == pytest.approx(-12.0)
    assert warp_mod._median_shift_semitones(130.81, 130.81) == pytest.approx(0.0, abs=1e-6)
    # Guards: any missing/zero/negative operand yields 0.0 (leave the take alone).
    assert warp_mod._median_shift_semitones(None, 220.0) == 0.0
    assert warp_mod._median_shift_semitones(110.0, None) == 0.0
    assert warp_mod._median_shift_semitones(0.0, 220.0) == 0.0
    assert warp_mod._median_shift_semitones(-5.0, 220.0) == 0.0


def test_score_median_note_hz():
    """The register target is the median of the score's pitched notes (rests skipped)."""
    from vox_tongue.render import note_to_hz_any

    score = compile_mod.compile(["I lay low"], ["A2", "A2", "A4"], bpm=120)
    hzs = [note_to_hz_any(s["note"]) for s in score["syllables"]
           if note_to_hz_any(s["note"])]
    assert warp_mod._score_median_note_hz(score) == pytest.approx(float(np.median(hzs)))


@whisper_gate
def test_warp_mode_full_vs_time_only_real(monkeypatch):
    """Real end-to-end mode comparison: full mode re-voices the take to the score note; time-only
    keeps the take near its native register. (Whisper- + say-gated; measures f0 with pyworld.)"""
    from vox_tongue.render import say_available

    if not say_available():
        pytest.skip("`say` unavailable")
    import pyworld as pw
    from vox_tongue.render import _say_render

    def _median_f0(x):
        xx = np.ascontiguousarray(x, dtype="float64")
        f0, t = pw.harvest(xx, SR, f0_floor=80.0, f0_ceil=700.0, frame_period=5.0)
        f0 = pw.stonemask(xx, f0, t, SR)
        v = f0[f0 > 0]
        return float(np.median(v)) if v.size else 0.0

    clip = _say_render("lay low", "Fred", SR)
    score = compile_mod.compile(["lay low"], ["A4", "A4"], bpm=90)  # 440 Hz target, far from `say`
    target = 440.0

    out_full, rep_full = warp_mod.warp_to_score(clip, SR, score, bpm=90, mode="full")
    out_to, rep_to = warp_mod.warp_to_score(clip, SR, score, bpm=90, mode="time-only")

    # full lands near the score note; time-only stays closer to the take's native register.
    assert abs(_median_f0(out_full) - target) < abs(_median_f0(out_to) - target)
    assert all(r["pitch_imposed"] for r in rep_full["syllables"])
    assert not any(r["pitch_imposed"] for r in rep_to["syllables"])
