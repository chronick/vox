"""Sustained-verse acceptance — the choral pipeline end to end on a 2-line mini.

Path: lyric packet (delivery sustained) -> tongue.compile_packet (even-spread, bpm 90) ->
tongue.render (say -> WORLD pitch) -> concat with breath gaps -> vox_larynx.world.harmonize.
This test proves the mini verse renders to nonzero audio of roughly the expected duration
and that the harmonize step records its voice count (the choir width).

    cd tools/vox-tongue && uv run pytest tests/test_sustained_verse.py -q
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from vox_lyric.packet import build_packet
from vox_tongue import render as render_mod
from vox_tongue.compile import compile_packet

SR = 44_100
BPM = 90.0
BREATH_S = 0.35
MELODY = ["A2", "C3", "E3", "D3", "C3"]

# A neutral 2-line mini; star marks a melisma syllable.
MINI = [("Slow river carry me home", "home"), ("Open water hold the line", "line")]


def _say_available():
    return render_mod.say_available()


def _norm(w):
    return re.sub(r"[^a-z]", "", w.lower())


def _apply_melisma(score, starred):
    """Double the dur of the starred word's (last) syllable; shift later syllables — the v0
    melisma approximation (one note/syllable schema, duration-only)."""
    syls = score["syllables"]
    idx = None
    for i, s in enumerate(syls):
        if _norm(s["word"]) == _norm(starred):
            idx = i
    if idx is None:
        return score
    extra = syls[idx]["dur_beats"]
    syls[idx]["dur_beats"] += extra
    for j in range(idx + 1, len(syls)):
        syls[j]["start_beat"] += extra
    return score


def _render_line(line, starred, melody):
    packet = build_packet([line], "sustained")
    score = compile_packet(packet, melody, bpm=BPM)
    _apply_melisma(score, starred)
    samples, manifest = render_mod.render(score, sr=SR, voice="Fred")
    return samples.astype("float64"), manifest, score


@pytest.mark.skipif(not _say_available(), reason="macOS `say` not available")
def test_sustained_mini_pipeline():
    from vox_larynx import world

    gap = np.zeros(int(BREATH_S * SR), dtype="float64")
    chunks, expected_beats = [], 0.0
    for li, (line, star) in enumerate(MINI):
        mel = MELODY[li % len(MELODY):] + MELODY[:li % len(MELODY)]  # rotate per line
        sig, manifest, score = _render_line(line, star, mel)
        # melisma doubled exactly one syllable's duration
        starred_syls = [s for s in score["syllables"] if _norm(s["word"]) == _norm(star)]
        assert starred_syls and starred_syls[-1]["dur_beats"] == pytest.approx(2.0)
        # the score's last onset+dur is the line's beat length
        last = score["syllables"][-1]
        expected_beats += last["start_beat"] + last["dur_beats"]
        chunks.append(sig)
        chunks.append(gap.copy())
    chunks.pop()  # no trailing gap
    dry = np.concatenate(chunks)

    # nonzero output
    assert dry.size > 0
    assert float(np.max(np.abs(dry))) > 0.0

    # duration ~= sum(line beat-lengths)*spb + one breath gap between the two lines (±20%)
    spb = 60.0 / BPM
    expected_s = expected_beats * spb + BREATH_S
    got_s = len(dry) / SR
    assert got_s == pytest.approx(expected_s, rel=0.20), f"{got_s:.2f}s vs ~{expected_s:.2f}s"

    # the harmonizer records its voice count (lead + chord(0,3,7) + drone = 5)
    mix, params = world.harmonize(dry, SR, chord=(0, 3, 7), mode="follow", drone=True)
    assert mix.size > 0
    assert params["voices"] == 5
    assert params["chord"] == [0, 3, 7]
    assert params["drone"] is True
