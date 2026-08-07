"""FLOW cadence-grammar tests. Compiler tests are exact + pure; render tests are gated on
`say`/`ffmpeg` (system binaries).

    cd tools/vox-flow && uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
from vox_flow import flow, render


def test_onset_times_are_grid_multiples():
    step = flow.step_seconds(140, 4)
    sc = flow.compile_flow("X.x.X.x.", bpm=140, grid=4)
    assert sc["n_onsets"] == 4
    times = [e["t"] for e in sc["events"]]
    assert times == pytest.approx([0 * step, 2 * step, 4 * step, 6 * step], abs=1e-4)
    assert [e["accent"] for e in sc["events"]] == [True, False, True, False]


def test_hold_extends_previous_onset():
    step = flow.step_seconds(120, 4)
    sc = flow.compile_flow("x---", bpm=120, grid=4)   # one onset held 3 extra steps
    assert sc["n_onsets"] == 1
    assert sc["events"][0]["dur"] == pytest.approx(4 * step, abs=1e-4)


def test_swing_delays_offbeats():
    plain = flow.compile_flow("xxxx", bpm=120, grid=4, swing=0.0)
    swung = flow.compile_flow("xxxx", bpm=120, grid=4, swing=0.5)
    # even steps unchanged; odd (off) steps pushed later
    assert swung["events"][0]["t"] == pytest.approx(plain["events"][0]["t"])
    assert swung["events"][1]["t"] > plain["events"][1]["t"]


def test_syllables_cycle_over_onsets():
    sc = flow.compile_flow("xxxxx", bpm=140, grid=4, syllables=["a", "b"])
    assert [e["syllable"] for e in sc["events"]] == ["a", "b", "a", "b", "a"]


def test_push_shifts_all_onsets():
    base = flow.compile_flow("x.x.", bpm=140, grid=4)
    drag = flow.compile_flow("x.x.", bpm=140, grid=4, push_ms=20.0)
    for b, d in zip(base["events"], drag["events"]):
        assert d["t"] == pytest.approx(b["t"] + 0.02, abs=1e-4)


def test_bad_swing_rejected():
    with pytest.raises(ValueError):
        flow.compile_flow("xxxx", swing=1.5)


def test_chain_registry_has_named_chains():
    assert set(flow.CHAINS) == {"grit", "bass"}
    for graph in flow.CHAINS.values():
        assert "agate" in graph and "alimiter" in graph


@pytest.mark.skipif(not (render.say_available() and render.ffmpeg_available()),
                    reason="say/ffmpeg not on PATH")
def test_render_and_grit_chain():
    sc = flow.compile_flow("X.x.X.x.", bpm=140, grid=4, syllables=["da", "ka", "ta", "ma"])
    dry = render.render_flow(sc, voice="Fred", pbas=90, sr=44100)
    assert dry.size > 0 and float(np.max(np.abs(dry))) > 0.01
    assert len(dry) / 44100 == pytest.approx(sc["bar_seconds"] + 0.25, abs=0.1)
    wet = render.apply_chain(dry, 44100, "grit")
    assert wet.size > 0 and float(np.max(np.abs(wet))) <= 0.95

    def rolloff95(x, sr=44100):
        S = np.abs(np.fft.rfft(x)); f = np.fft.rfftfreq(len(x), 1 / sr); c = np.cumsum(S)
        return float(f[np.searchsorted(c, 0.95 * c[-1])])

    assert rolloff95(wet) < rolloff95(dry)  # the grit chain band-limits
