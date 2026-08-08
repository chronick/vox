"""Tests for vox tongue — the phoneme DSL, compile placement, schema round-trip, and the
deterministic say->WORLD render path (the per-syllable pitch check is the acceptance test).

Run (siblings on PYTHONPATH):

    cd tools/vox-tongue && uv run pytest -q
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest
from vox_tongue import compile as compile_mod
from vox_tongue import cli, g2p, schema

SR = 44_100


def test_say_render_rejects_empty_audio_before_world(monkeypatch):
    import numpy as np
    import soundfile as sf
    from vox_tongue import render

    def fake_run(cmd, capture_output):
        Path(cmd[cmd.index("-o") + 1]).touch()
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(render.subprocess, "run", fake_run)
    monkeypatch.setattr(sf, "read", lambda *_args, **_kwargs: (np.zeros((0, 1)), SR))

    with pytest.raises(RuntimeError, match="say produced no audio"):
        render._say_render("ma", "Fred", SR)


# ---------------------------------------------------------------------------
# g2p — the phone contract.
# ---------------------------------------------------------------------------
def test_g2p_machine_two_syllables_with_stress():
    phones = g2p.word_phones("machine")
    assert phones == ["M", "AH0", "SH", "IY1", "N"]
    syls = g2p.split_syllables(phones)
    assert syls == [["M", "AH0"], ["SH", "IY1", "N"]]  # onsets attach forward, trailing N appends


def test_g2p_unknown_word_is_null():
    assert g2p.word_phones("zzxqwv") is None
    line = g2p.syllabify_line("zzxqwv")
    assert line[0]["syllable_phones"] is None  # OOV -> null phones (consumers handle null)
    assert line[0]["syllable_texts"]  # still yields pronounceable text fragments


def test_syllabify_line_text_fragments():
    [entry] = g2p.syllabify_line("machine")
    assert entry["word"] == "machine"
    assert entry["syllable_texts"] == ["ma", "chine"]  # split at the vowel-group boundary
    assert entry["syllable_phones"] == [["M", "AH0"], ["SH", "IY1", "N"]]


# ---------------------------------------------------------------------------
# compile — melody cycling + placement.
# ---------------------------------------------------------------------------
def test_compile_cycles_melody_over_four_syllables():
    score = compile_mod.compile(["machine machine"], ["A2", "C3"], bpm=120)
    assert len(score["syllables"]) == 4  # 2 words x 2 syllables
    assert [s["note"] for s in score["syllables"]] == ["A2", "C3", "A2", "C3"]


def test_compile_flow_placement_matches_compile_flow():
    from vox_flow.flow import compile_flow

    lines = ["machine machine"]
    melody = ["A2", "C3"]
    bpm = 120.0
    pattern = "X.x.X.x."
    score = compile_mod.compile(lines, melody, bpm=bpm, flow_pattern=pattern)

    atoms = [t for e in g2p.syllabify_line(lines[0]) for t in e["syllable_texts"]]
    flow_score = compile_flow(pattern, bpm=bpm, syllables=atoms)
    events = flow_score["events"]
    spb = 60.0 / bpm

    assert len(score["syllables"]) == len(events)
    for syl, ev in zip(score["syllables"], events):
        assert syl["start_beat"] == pytest.approx(ev["t"] / spb)
        assert syl["dur_beats"] == pytest.approx(ev["dur"] / spb)


def test_compile_packet_cli_reads_file_and_writes_score(tmp_path):
    from vox_lyric.packet import build_packet

    packet_path = tmp_path / "river-packet.json"
    score_path = tmp_path / "river.yaml"
    packet_path.write_text(json.dumps(build_packet(["slow river carry me home"], "sustained")))

    rc = cli.main([
        "compile-packet", "--packet", str(packet_path),
        "--melody", "A2,C3,E3,D3,C3", "--bpm", "90", "--score", str(score_path),
    ])

    assert rc == 0
    score = schema.load_score(str(score_path))
    assert score["meta"]["bpm"] == 90.0
    assert score["syllables"][0]["note"] == "A2"


def test_compile_packet_cli_reads_stdin(monkeypatch, capsys):
    from vox_lyric.packet import build_packet

    packet = build_packet(["slow river carry me home"], "sustained")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(packet)))

    rc = cli.main(["compile-packet", "--melody", "A2,C3", "--bpm", "90"])

    assert rc == 0
    score = schema.load_score(schema.from_yaml(capsys.readouterr().out))
    assert score["meta"]["bpm"] == 90.0
    assert len(score["syllables"]) == packet["lines"][0]["syllable_count"]


def test_compile_packet_cli_reports_rejected_lyrics(tmp_path, capsys):
    from vox_lyric.packet import build_packet

    packet_path = tmp_path / "rejected.json"
    packet_path.write_text(json.dumps(build_packet(["darkness eternal soul"], "percussive")))

    rc = cli.main([
        "compile-packet", "--packet", str(packet_path), "--melody", "E2,G2",
    ])

    assert rc == 2
    err = capsys.readouterr().err
    assert "lyric packet rejected" in err
    assert "rewrite" in err


# ---------------------------------------------------------------------------
# schema — YAML round-trip.
# ---------------------------------------------------------------------------
def test_schema_yaml_round_trip():
    score = compile_mod.compile(["wake the machine"], ["A2", "C3", "E3"], bpm=128)
    text = schema.to_yaml(score)
    back = schema.load_score(schema.from_yaml(text))
    assert back["meta"]["bpm"] == 128.0
    assert len(back["syllables"]) == len(score["syllables"])
    assert [s["note"] for s in back["syllables"]] == [s["note"] for s in score["syllables"]]
    assert [s["phones"] for s in back["syllables"]] == [s["phones"] for s in score["syllables"]]


def test_schema_rejects_bad_phone():
    bad = {"meta": {"bpm": 120.0},
           "syllables": [{"text": "ma", "word": "machine", "phones": ["not a phone"],
                          "start_beat": 0.0, "dur_beats": 1.0, "note": "A2", "dyn": 1.0,
                          "articulation": None}]}
    with pytest.raises(ValueError):
        schema.load_score(bad)


# ---------------------------------------------------------------------------
# render — the acceptance test: per-syllable WORLD pitch imposition.
# ---------------------------------------------------------------------------
def _say_available():
    from vox_tongue import render as render_mod

    return render_mod.say_available()


@pytest.mark.skipif(not _say_available(), reason="macOS `say` not available")
def test_render_per_syllable_pitch():
    from vox_larynx import world
    from vox_tongue import render as render_mod

    bpm = 100.0
    score = compile_mod.compile(["machine"], ["A2", "E2"], bpm=bpm)
    targets = [world.note_to_hz("A2"), world.note_to_hz("E2")]
    assert targets[0] == pytest.approx(110.0, abs=0.5)
    assert targets[1] == pytest.approx(82.4, abs=0.5)

    samples, manifest = render_mod.render(score, sr=SR, voice="Fred")
    assert len(manifest) == 2

    for seg_info, target in zip(manifest, targets):
        start = int(seg_info["t"] * SR)
        end = int((seg_info["t"] + seg_info["dur"]) * SR)
        seg = samples[start:end].astype("float64")
        f0, *_ = world.analyze(seg, SR)
        median = world.f0_stats(f0)["f0_median_hz"]
        assert median is not None, f"segment {seg_info['text']!r} came out unvoiced"
        rel = abs(median - target) / target
        assert rel < 0.08, f"{seg_info['text']!r}: median {median} Hz vs target {target} Hz ({rel:.2%})"
