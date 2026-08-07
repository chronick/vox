"""Tests for the DiffSinger ``.ds`` emitter (vox_tongue.ds).

The .ds format is pure data, so these tests are audio-free and fast. Run:

    cd tools/vox-tongue && uv run pytest tests/test_ds.py -q
"""

from __future__ import annotations

import json

import pytest
from vox_tongue import compile as compile_mod
from vox_tongue import ds, g2p

# A neutral acceptance corpus for phone coverage: ordinary cmudict words chosen for wide
# vowel/consonant variety (measure/vision cover ZH, coin covers OY, book covers UH, and so
# on). Every word resolves in cmudict, so these lines exercise the ARPABET->DS map end-to-end.
SUSTAINED_LINES = [
    "Slow river carry me home",
    "Open water hold the line",
    "Every morning light arrives",
    "I follow the sound of the tide",
    "A measure of vision and time",
    "The moon is a coin on the water",
    "Take the good book down",
    "Read it slow by the fire",
    "How the round vowels open wide",
    "Away we go over the mountain",
    "Nothing here is out of tune",
    "The choir holds one long note",
]
PERCUSSIVE_LINES = [
    "Kick the pattern back to the top",
    "Cut the deck and count to ten",
    "Punch the clock and check the tape",
    "Judge the change by the chart",
    "Grab a thing and get it done",
    "Stack the bricks up brick by brick",
    "Quick tricks click in the mix",
    "Drop the beat and pick it up",
    "Thick fog sits on the dock",
    "Big trucks bring the boxes in",
    "Snap the strap and pack the bag",
    "Push the button start again",
]


def _fields(seg):
    return (seg["ph_seq"].split(), seg["ph_dur"].split(), seg["ph_num"].split(),
            seg["note_seq"].split(), seg["note_dur"].split())


# ---------------------------------------------------------------------------
# a 2-word score emits a valid .ds
# ---------------------------------------------------------------------------
def test_two_word_score_emits_valid_ds():
    score = compile_mod.compile(["wake machine"], ["A2", "C3"], bpm=120)
    seg = ds.to_ds(score)

    # parseable as JSON when wrapped as a real .ds file (array of segments)
    parsed = json.loads(json.dumps([seg]))
    assert isinstance(parsed, list) and isinstance(parsed[0], dict)

    ph_seq, ph_dur, ph_num, note_seq, note_dur = _fields(seg)
    # ph_seq / ph_dur are parallel
    assert len(ph_seq) == len(ph_dur)
    # note-level arrays are parallel and ph_num sums to the phone count
    assert len(note_seq) == len(note_dur) == len(ph_num) == len(seg["note_slur"].split())
    assert sum(int(n) for n in ph_num) == len(ph_seq)

    # SP silences: leading, trailing, and one between the two words
    assert ph_seq[0] == "SP" and ph_seq[-1] == "SP"
    assert ph_seq.count("SP") >= 3


def test_durations_sum_to_the_score_total():
    bpm = 120.0
    score = compile_mod.compile(["wake machine"], ["A2", "C3"], bpm=bpm)
    seg = ds.to_ds(score)
    _, ph_dur, ph_num, _, note_dur = _fields(seg)

    syls = score["syllables"]
    spb = 60.0 / bpm
    expected_total = max(s["start_beat"] + s["dur_beats"] for s in syls) * spb

    assert sum(float(d) for d in ph_dur) == pytest.approx(expected_total, abs=1e-4)
    # note_dur totals to the same timeline
    assert sum(float(d) for d in note_dur) == pytest.approx(expected_total, abs=1e-4)
    # each note's duration equals the sum of its own phones' durations (internal consistency)
    ph_dur_f = [float(d) for d in seg["ph_dur"].split()]
    idx = 0
    for k, n in enumerate(int(x) for x in ph_num):
        chunk = sum(ph_dur_f[idx:idx + n])
        assert chunk == pytest.approx(float(note_dur[k]), abs=1e-4)
        idx += n


def test_f0_curve_present_and_flat_per_note():
    # start at beat 1 so there is a real leading rest (pre-roll) -> unvoiced frames to check
    score = {"meta": {"bpm": 100.0},
             "syllables": [
                 {"text": "ma", "word": "machine", "phones": ["M", "AH0"], "start_beat": 1.0,
                  "dur_beats": 1.0, "note": "A2", "dyn": 1.0, "articulation": None},
                 {"text": "chine", "word": "machine", "phones": ["SH", "IY1", "N"], "start_beat": 2.0,
                  "dur_beats": 1.0, "note": "C3", "dyn": 1.0, "articulation": None},
             ]}
    seg = ds.to_ds(score, with_f0=True, f0_timestep=0.024)
    assert "f0_seq" in seg and "f0_timestep" in seg
    f0 = [float(x) for x in seg["f0_seq"].split()]
    # the voiced frames should include the two target pitches (flat per note)
    voiced = {round(v, 1) for v in f0 if v > 0}
    assert round(ds.note_to_hz("A2"), 1) in voiced
    assert round(ds.note_to_hz("C3"), 1) in voiced
    # the leading rest is unvoiced (0 Hz)
    assert any(v == 0.0 for v in f0)


def test_no_f0_flag_omits_curve():
    score = compile_mod.compile(["machine"], ["A2"], bpm=100)
    seg = ds.to_ds(score, with_f0=False)
    assert "f0_seq" not in seg and "f0_timestep" not in seg


def test_note_seq_normalizes_flats_and_bare_hz():
    # a note-name flat -> sharp spelling; a bare Hz -> nearest name, exact Hz in f0
    score = {"meta": {"bpm": 120.0},
             "syllables": [
                 {"text": "a", "word": "a", "phones": ["AH0"], "start_beat": 0.0,
                  "dur_beats": 1.0, "note": "Eb3", "dyn": 1.0, "articulation": None},
                 {"text": "b", "word": "b", "phones": ["B", "IY1"], "start_beat": 1.0,
                  "dur_beats": 1.0, "note": 220.0, "dyn": 1.0, "articulation": None},
             ]}
    seg = ds.to_ds(score)
    notes = [n for n in seg["note_seq"].split() if n != "rest"]
    assert notes == ["D#3", "A3"]  # Eb3 -> D#3 ; 220 Hz -> A3
    assert round(ds.note_to_hz("A3"), 1) == 220.0


def test_oov_syllable_raises():
    score = {"meta": {"bpm": 120.0},
             "syllables": [{"text": "zzz", "word": "zzz", "phones": None, "start_beat": 0.0,
                            "dur_beats": 1.0, "note": "A2", "dyn": 1.0, "articulation": None}]}
    with pytest.raises(ValueError):
        ds.to_ds(score)


# ---------------------------------------------------------------------------
# ARPABET -> DS mapping coverage
# ---------------------------------------------------------------------------
def test_mapping_covers_full_cmudict_phone_inventory():
    """Every phone ``pronouncing`` can emit (stress-stripped) is in ARPABET_TO_DS."""
    # The canonical CMU ARPABET inventory: 15 vowels + 24 consonants.
    cmu = {
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
        "OW", "OY", "UH", "UW",
        "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG",
        "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
    }
    assert len(cmu) == 39
    assert cmu.issubset(set(ds.ARPABET_TO_DS))
    # and the map has no phantom keys
    assert set(ds.ARPABET_TO_DS) == cmu


def test_no_unmapped_phones_in_lyric_corpus():
    """Every phone produced by the fixture corpus maps cleanly."""
    unmapped = set()
    for line in SUSTAINED_LINES + PERCUSSIVE_LINES:
        for entry in g2p.syllabify_line(line):
            groups = entry["syllable_phones"]
            if groups is None:  # OOV word — separate concern
                continue
            for grp in groups:
                for ph in grp:
                    base = ds.strip_stress(ph)
                    if base not in ds.ARPABET_TO_DS:
                        unmapped.add(ph)
    assert unmapped == set(), f"unmapped phones in lyric corpus: {sorted(unmapped)}"


def test_full_verse_score_emits_end_to_end():
    """A whole verse compiles + emits a valid .ds with no unmapped phone / OOV error."""
    score = compile_mod.compile(SUSTAINED_LINES, ["A3", "C4", "E4", "D4"], bpm=72)
    seg = ds.to_ds(score)
    ph_seq = seg["ph_seq"].split()
    # every non-SP phone is a valid DS phoneme
    ds_vocab = set(ds.ARPABET_TO_DS.values()) | {"SP"}
    assert all(p in ds_vocab for p in ph_seq)
    assert sum(int(n) for n in seg["ph_num"].split()) == len(ph_seq)


def test_map_phone_strips_stress_and_lowercases():
    assert ds.map_phone("IY1") == "iy"
    assert ds.map_phone("AH0") == "ah"
    assert ds.map_phone("NG") == "ng"
    with pytest.raises(ValueError):
        ds.map_phone("XQ")
