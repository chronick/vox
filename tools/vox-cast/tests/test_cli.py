"""CLI behavior: degradation hints, info, setup --status, and the full pipe
through the fake engine (real runner, stub ML)."""

import hashlib
import json

import pytest

from conftest import frames_bytes, make_wav, parse_frames, run_cli


def test_convert_without_engine_degrades_with_hint(tmp_path, cast_tree):
    wav = make_wav(tmp_path / "take.wav")
    r = run_cli(["convert", "--in", str(wav), "--model", "growl"])
    assert r.returncode == 1
    assert "vox cast setup" in r.stderr.decode()
    err = [f for f in parse_frames(r.stdout) if f.get("kind") == "error"]
    assert err and err[-1]["data"]["code"] == "op_failed"
    assert "engine not installed" in err[-1]["data"]["message"]


def test_convert_unknown_model_degrades(tmp_path, cast_tree):
    wav = make_wav(tmp_path / "take.wav")
    r = run_cli(["convert", "--in", str(wav), "--model", "whisper"])
    assert r.returncode == 1
    assert "available: growl" in r.stderr.decode()
    err = [f for f in parse_frames(r.stdout) if f.get("kind") == "error"]
    assert err and err[-1]["data"]["code"] == "not_found"


def test_pipe_convert_through_fake_engine(tmp_path, cast_tree, fake_engine):
    wav = make_wav(tmp_path / "take.wav", seconds=0.3, sr=22050)
    stdin, src = frames_bytes(wav)
    r = run_cli([
        "convert", "--model", "growl", "--pitch", "3", "--trust-model",
    ], stdin_bytes=stdin)
    assert r.returncode == 0, r.stderr.decode()

    frames = parse_frames(r.stdout)
    assert frames[0]["id"] == src["id"], "input frame must pass through"
    derived = frames[-1]
    assert derived["kind"] == "audio"
    assert derived["role"] == "voice.wet"
    assert derived["of"] == src["id"]
    assert derived["lineage"] == [src["id"]]
    assert derived["op"] == "cast-convert"
    assert derived["params"]["model"] == "growl"
    assert derived["params"]["pitch"] == 3
    # The stub engine copies audio verbatim, so the canonical-PCM hash dedupes
    # to the same CAS object — the derived FRAME is still new (id/op/params).
    assert derived["hash"] == src["hash"]
    assert derived["id"] != src["id"]

    from smplstream import cas
    assert cas.get_path(derived["hash"]).exists()

    rec = json.loads(fake_engine["record"].read_text())
    assert rec["model_path"].endswith("growl.pth")
    assert rec["index_path"].endswith("growl.index")
    assert rec["params"]["f0up_key"] == 3


def test_standalone_in_file_closes_lineage(tmp_path, cast_tree, fake_engine):
    wav = make_wav(tmp_path / "take.wav")
    r = run_cli([
        "convert", "--in", str(wav), "--model", "growl", "--trust-model",
    ])
    assert r.returncode == 0, r.stderr.decode()
    frames = parse_frames(r.stdout)
    audio = [f for f in frames if f.get("kind") == "audio"]
    assert len(audio) == 2, "source ingest frame + derived frame"
    src, derived = audio
    assert derived["of"] == src["id"]
    assert src["op"] == "from-file"


def test_info_reports_model_from_disk(cast_tree):
    r = run_cli(["info", "--model", "growl"])
    assert r.returncode == 0
    info = json.loads(r.stdout)
    assert info["name"] == "growl"
    assert info["sample_rate"] == 40000
    assert info["pth"]["sha256"] == hashlib.sha256(b"\x00" * 2048).hexdigest()
    assert info["index"]["sha256"] == hashlib.sha256(b"\x00" * 1024).hexdigest()
    assert {sidecar["file"] for sidecar in info["sidecars"]} == {
        "config.json", "model_info.json",
    }


def test_info_direct_pth_does_not_inventory_unrelated_files(tmp_path):
    pth = tmp_path / "weights.pth"
    pth.write_bytes(b"weights")
    (tmp_path / "MODEL_CARD.md").write_text("known provenance")
    (tmp_path / "unrelated.json").write_text("another download")

    r = run_cli(["info", "--model", str(pth)])

    assert r.returncode == 0
    info = json.loads(r.stdout)
    assert [sidecar["file"] for sidecar in info["sidecars"]] == ["MODEL_CARD.md"]


def test_installed_engine_requires_explicit_model_trust(tmp_path, cast_tree, fake_engine):
    wav = make_wav(tmp_path / "take.wav")
    r = run_cli(["convert", "--in", str(wav), "--model", "growl"])

    assert r.returncode == 1
    message = r.stderr.decode()
    assert ".pth models are Python pickle files" in message
    assert "venv is not a security boundary" in message
    assert "--trust-model" in message
    assert not fake_engine["record"].exists()


@pytest.mark.parametrize(("option", "value"), [
    ("--index-rate", "nan"),
    ("--index-rate", "1.01"),
    ("--index-rate", "-0.01"),
    ("--rms-mix", "inf"),
    ("--rms-mix", "1.01"),
    ("--protect", "-inf"),
    ("--protect", "0.51"),
    ("--f0-method", "dio"),
    ("--device", "cuda"),
    ("--device", "cpu:-1"),
    ("--device", "mps:0"),
])
def test_convert_rejects_unsafe_parameters(option, value):
    r = run_cli(["convert", "--model", "growl", option, value])

    assert r.returncode == 2
    assert b"error:" in r.stderr


@pytest.mark.parametrize(("option", "value"), [
    ("--index-rate", "0"),
    ("--index-rate", "1"),
    ("--rms-mix", "0"),
    ("--rms-mix", "1"),
    ("--protect", "0"),
    ("--protect", "0.5"),
    ("--f0-method", "rmvpe"),
    ("--f0-method", "harvest"),
    ("--f0-method", "crepe"),
    ("--device", "cpu"),
    ("--device", "cpu:12"),
    ("--device", "cuda:0"),
    ("--device", "mps"),
])
def test_convert_accepts_parameter_boundaries(option, value, cast_tree):
    r = run_cli(["convert", "--model", "growl", option, value], stdin_bytes=b"")

    assert r.returncode == 1
    assert b"no resolvable audio" in r.stderr


def test_import_directory_and_list_preserve_sidecars(tmp_path):
    source = tmp_path / "downloaded-voice"
    source.mkdir()
    (source / "weights.pth").write_bytes(b"weights")
    (source / "voice.index").write_bytes(b"index")
    (source / "config.json").write_text(json.dumps({"data": {"sample_rate": 48000}}))
    (source / "portrait.png").write_bytes(b"png")
    (source / "training-audio.wav").write_bytes(b"do not copy training data")

    imported = run_cli(["import", "--model", str(source), "--name", "alto"])

    assert imported.returncode == 0, imported.stderr.decode()
    info = json.loads(imported.stdout)
    assert info["name"] == "alto"
    assert info["pth"]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert info["index"]["sha256"] == hashlib.sha256(b"index").hexdigest()
    assert {sidecar["file"] for sidecar in info["sidecars"]} == {
        "config.json", "portrait.png",
    }
    destination = tmp_path / "casts" / "alto"
    assert (destination / "weights.pth").read_bytes() == b"weights"
    assert (destination / "voice.index").read_bytes() == b"index"
    assert (destination / "config.json").is_file()
    assert (destination / "portrait.png").is_file()
    assert not (destination / "training-audio.wav").exists()

    listed = run_cli(["list"])
    assert listed.returncode == 0
    payload = json.loads(listed.stdout)
    assert [cast["name"] for cast in payload["casts"]] == ["alto"]
    assert payload["casts"][0]["sample_rate"] == 48000
    assert "sha256" not in payload["casts"][0]["pth"]
    assert "sha256" not in payload["casts"][0]["index"]


def test_import_pth_with_explicit_index_and_refuses_overwrite(tmp_path):
    source = tmp_path / "weights.pth"
    index = tmp_path / "chosen.index"
    source.write_bytes(b"weights")
    index.write_bytes(b"index")
    (tmp_path / "MODEL_CARD.md").write_text("authorized synthetic cast")
    (tmp_path / "LICENSE").write_text("model license")
    (tmp_path / "unrelated.json").write_text("do not sweep a downloads folder")

    first = run_cli([
        "import", "--model", str(source), "--index", str(index), "--name", "soprano",
    ])
    second = run_cli([
        "import", "--model", str(source), "--index", str(index), "--name", "soprano",
    ])

    assert first.returncode == 0, first.stderr.decode()
    assert json.loads(first.stdout)["name"] == "soprano"
    destination = tmp_path / "casts" / "soprano"
    assert (destination / "MODEL_CARD.md").is_file()
    assert (destination / "LICENSE").is_file()
    assert not (destination / "unrelated.json").exists()
    assert second.returncode == 1
    assert "refusing to overwrite" in second.stderr.decode()


def test_setup_status_reports_absent_engine(tmp_path):
    r = run_cli(["setup", "--status"])
    assert r.returncode == 0
    st = json.loads(r.stdout)
    assert st["installed"] is False
    assert st["dir"].endswith("engine")


def test_no_audio_input_errors(cast_tree):
    r = run_cli(["convert", "--model", "growl"], stdin_bytes=b"")
    assert r.returncode == 1
    assert "no resolvable audio" in r.stderr.decode()
