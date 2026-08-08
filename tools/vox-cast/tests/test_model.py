"""Cast resolution: the three --model spellings and their failure messages."""

import json

import pytest
from vox_cast import model


def test_resolve_direct_pth(cast_tree):
    r = model.resolve_cast(str(cast_tree / "growl.pth"))
    assert r["name"] == "growl"
    assert r["pth"].name == "growl.pth"
    assert r["index"].name == "growl.index"


def test_resolve_dir(cast_tree):
    r = model.resolve_cast(str(cast_tree))
    assert r["name"] == "growl"
    assert r["dir"] == cast_tree


def test_resolve_library_name(cast_tree):
    r = model.resolve_cast("growl")
    assert r["pth"] == cast_tree / "growl.pth"


def test_unknown_name_lists_available(cast_tree):
    with pytest.raises(model.CastNotFound, match="available: growl"):
        model.resolve_cast("whisper")


def test_dir_with_two_pths_rejected(tmp_path):
    d = tmp_path / "multi"
    d.mkdir()
    (d / "a.pth").write_bytes(b"x")
    (d / "b.pth").write_bytes(b"x")
    with pytest.raises(model.CastNotFound, match="a.pth, b.pth"):
        model.resolve_cast(str(d))


def test_two_indexes_need_explicit_pick(tmp_path):
    d = tmp_path / "twoidx"
    d.mkdir()
    (d / "m.pth").write_bytes(b"x")
    (d / "a.index").write_bytes(b"x")
    (d / "b.index").write_bytes(b"x")
    with pytest.raises(model.CastNotFound, match="--index"):
        model.resolve_cast(str(d))
    r = model.resolve_cast(str(d), index=str(d / "b.index"))
    assert r["index"].name == "b.index"


def test_non_pth_file_rejected(tmp_path):
    f = tmp_path / "weights.onnx"
    f.write_bytes(b"x")
    with pytest.raises(model.CastNotFound, match="not a .pth"):
        model.resolve_cast(str(f))


def test_cast_info_reads_sidecar_metadata(cast_tree):
    info = model.cast_info(model.resolve_cast("growl"))
    assert info["sample_rate"] == 40000
    assert info["model_info"]["embedder_model"] == "contentvec"
    assert info["pth"]["file"] == "growl.pth"
    assert info["index"]["mb"] == pytest.approx(0.0, abs=0.1)
    json.dumps(info)  # must stay serializable
