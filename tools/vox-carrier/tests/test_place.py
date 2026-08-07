"""Line-placement assembler tests (``place_clips_on_grid``).

TTS-independent: synthetic clips only (DC blocks + sine bursts), so the assembler's contract is
pinned without any TTS/vocoder deps. Per-line clips must drop onto the 2-bar grid *without*
syllable chopping — the structural guarantees are (a) each line STARTS on its bar boundary and
(b) seams never click, even when a clip over-runs its slot or falls short of it.

    cd tools/vox-carrier && uv run pytest tests/test_place.py -q
"""

from __future__ import annotations

import numpy as np
from vox_carrier.carrier import place_clips_on_grid

BPM = 142
SR = 44100
BARS = 2
BEATS = 4


def _slot() -> int:
    return round(BARS * BEATS * 60.0 / BPM * SR)


def test_starts_land_on_two_bar_boundaries():
    slot = _slot()
    # three short DC clips, each well under a 2-bar slot (~3.38 s) → gaps between them.
    clips = [np.full(int(0.5 * SR), 0.8, dtype="float32") for _ in range(3)]
    out, starts = place_clips_on_grid(clips, SR, BPM, bars_per_line=BARS, beats_per_bar=BEATS)

    assert starts == [0, slot, 2 * slot]
    # onset of each clip is at its start: energy just after start, silence just before it.
    for i, s in enumerate(starts):
        assert abs(out[s + 100]) > 0.1, f"clip {i} should carry energy at its boundary"
        if i > 0:
            assert abs(out[s - 200]) < 1e-4, f"gap before clip {i} should be silent"


def test_short_clips_leave_a_silent_gap():
    slot = _slot()
    clips = [np.full(int(0.5 * SR), 0.8, dtype="float32") for _ in range(2)]
    out, starts = place_clips_on_grid(clips, SR, BPM)
    # region strictly between end of clip 0 and start of clip 1 is pure silence (a natural breath).
    gap = out[int(0.5 * SR) + 500: slot - 500]
    assert gap.size > 0 and float(np.max(np.abs(gap))) < 1e-6


def test_no_click_at_seams_even_on_overrun():
    # DC blocks click hard at raw edges (0 -> 0.9 step). Make clip 0 LONGER than a slot so its tail
    # overlaps clip 1's head; the edge fades must keep the overlap-add continuous.
    slot = _slot()
    long_clip = np.full(slot + int(0.4 * SR), 0.9, dtype="float32")   # over-runs into next line
    short_clip = np.full(int(0.6 * SR), -0.9, dtype="float32")
    out, starts = place_clips_on_grid([long_clip, short_clip], SR, BPM, fade_ms=4.0)

    # first sample-difference bound: with a 4 ms fade a DC edge ramps at ~1/(0.004*sr) per sample.
    max_step = float(np.max(np.abs(np.diff(out.astype("float64")))))
    assert max_step < 0.05, f"seam click detected: max |Δ| = {max_step:.4f}"
    # output length is the farthest-reaching clip: clip 1 ends at slot + its length, past clip 0's
    # over-run (slot + 0.4 s) since 0.6 s > 0.4 s.
    assert out.size == slot + short_clip.size
    assert starts == [0, slot]


def test_sine_clips_placed_without_amplitude_blowup():
    slot = _slot()
    t = np.arange(int(1.0 * SR)) / SR
    clips = [np.sin(2 * np.pi * 110 * t).astype("float32") for _ in range(4)]
    out, starts = place_clips_on_grid(clips, SR, BPM)
    assert starts == [0, slot, 2 * slot, 3 * slot]
    # non-overlapping sines: peak stays ~1.0, never sums to >1 (proves no accidental stacking).
    assert float(np.max(np.abs(out))) <= 1.001
