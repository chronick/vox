"""CLI behavior: degradation hints, info, setup --status, and the full pipe
through the fake engine (real runner, stub ML)."""

import json

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
    r = run_cli(["convert", "--model", "growl", "--pitch", "3"], stdin_bytes=stdin)
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
    r = run_cli(["convert", "--in", str(wav), "--model", "growl"])
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
