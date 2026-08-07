"""Take-card orchestration integration test — render → measure → verify.

    cd tools/vox-take && uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from vox_take import orchestrate

SR = 44_100


def _write_vowel(path, f0=200.0, dur=1.2):
    from scipy.signal import lfilter
    n = int(SR * dur)
    rng = np.random.default_rng(0)
    src = np.zeros(n)
    phase = 0.0
    for i in range(n):
        phase += f0 * (1 + 0.004 * rng.standard_normal()) / SR
        if phase >= 1.0:
            phase -= 1.0
            src[i] = 1.0
    out = np.zeros(n)
    for fc, bw in [(730, 80), (1090, 90), (2440, 120)]:
        r = np.exp(-np.pi * bw / SR); th = 2 * np.pi * fc / SR
        out += lfilter([1 - r], [1.0, -2 * r * np.cos(th), r * r], src)
    sf.write(path, (out / np.max(np.abs(out)) * 0.9).astype("float32"), SR)


def test_harmonize_take_renders_and_measures(tmp_path):
    src = tmp_path / "lead.wav"
    _write_vowel(str(src))
    card = {"name": "t", "source": str(src),
            "render": {"op": "harmonize", "params": {"chord": [0, 3, 7], "drone": True}},
            "target": {"breathiness": 0.1, "roughness": 0.1, "spatiality": 0.2}}
    report = orchestrate.run_take(card)
    assert set(report["measured_vector"]) == {"humanness", "breathiness", "roughness",
                                             "intelligibility", "multiplicity", "spatiality"}
    assert report["render"]["voices"] == 5
    # the self-verifying number exists for the axes present in both target and measurement
    assert report["verification"]["mean_abs_error"] is not None
    assert "breathiness" in report["verification"]["per_axis_error"]
    # audio came back
    assert report["audio"].size > 0 and report["sr"] == SR


def test_render_op_pitch_imposition(tmp_path):
    src = tmp_path / "lead.wav"
    _write_vowel(str(src), f0=200.0)
    card = {"name": "p", "source": str(src),
            "render": {"op": "render", "params": {"to_hz": 300.0}}}
    report = orchestrate.run_take(card)
    # a monotone-at-300 render should measure near 300 Hz
    assert report["measured_vector"] is not None
    assert report["render"]["op"] == "render"


def test_missing_source_errors():
    with pytest.raises(ValueError):
        orchestrate.render_take({"render": {"op": "harmonize"}})
