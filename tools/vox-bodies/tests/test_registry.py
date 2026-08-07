"""Body-registry tests — palette validation, fingerprint measurement, SC-gated render.

    cd tools/vox-bodies && uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
from vox_bodies import registry


def test_shipped_palette_loads_and_validates():
    bodies = registry.load_bodies()
    names = {b["name"] for b in bodies}
    assert {"growl-55", "subsaw-55", "throat-60", "fof-a-180", "fof-impossible"} <= names
    for b in bodies:
        assert b["engine"] in registry.VALID_ENGINES
        assert isinstance(b.get("tags", []), list)


def test_bad_engine_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("bodies:\n- name: x\n  engine: modem\n")
    with pytest.raises(ValueError, match="unknown engine"):
        registry.load_bodies(p)


def test_missing_name_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("bodies:\n- engine: fof\n")
    with pytest.raises(ValueError, match="missing required field"):
        registry.load_bodies(p)


def test_duplicate_name_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("bodies:\n- name: a\n  engine: fof\n- name: a\n  engine: fof\n")
    with pytest.raises(ValueError, match="duplicate"):
        registry.load_bodies(p)


def test_bad_tags_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("bodies:\n- name: a\n  engine: fof\n  tags: nope\n")
    with pytest.raises(ValueError, match="tags must be a list"):
        registry.load_bodies(p)


def test_measure_body_fingerprints_synthetic():
    sr = 44100
    t = np.arange(int(sr * 1.5)) / sr
    y = sum(np.sin(2 * np.pi * 110.0 * k * t) / k for k in range(1, 12)) * 0.5
    fp = registry.measure_body(np.asarray(y), sr, target_hint=110.0)
    assert fp["f0_hz"] == pytest.approx(110.0, abs=3.0)
    assert fp["centroid_hz"] is not None and fp["rolloff_hz"] is not None


def _sc_available() -> bool:
    try:
        from smpl_synth.backends import sc_available
        return sc_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _sc_available(), reason="SuperCollider not on PATH")
def test_render_sc_body_growl_lands_in_bass_band():
    y, sr = registry.render_body("growl-55", dur=2.0)
    assert y.ndim == 1 and len(y) > sr  # ~2 s of mono audio
    assert np.max(np.abs(y)) > 0.0
    fp = registry.measure_body(np.asarray(y), sr, target_hint=55.0)
    assert fp["f0_hz"] is not None and 45.0 <= fp["f0_hz"] <= 90.0
