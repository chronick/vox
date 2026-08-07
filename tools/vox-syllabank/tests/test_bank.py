"""Tests for the syllabank — schema gate, round-trip, lookup ladder, seed smoke.

    cd tools/vox-syllabank && uv run pytest -q
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from vox_syllabank import bank as B

SR = 44_100


# ---------------------------------------------------------------------------
# Phone contract.
# ---------------------------------------------------------------------------
def test_syllabify_machine():
    assert B.syllabify(["M", "AH0", "SH", "IY1", "N"]) == [["M", "AH0"], ["SH", "IY1", "N"]]


def test_syllabify_consonants_attach_forward_and_trail():
    # leading consonant cluster + trailing consonants with no following vowel
    assert B.syllabify(["S", "IH1", "L", "T"]) == [["S", "IH1", "L", "T"]]
    assert B.syllabify(["K", "AE1", "T"]) == [["K", "AE1", "T"]]


def test_is_vowel():
    assert B.is_vowel("IY1") and B.is_vowel("AH0") and B.is_vowel("AE2")
    assert not B.is_vowel("SH") and not B.is_vowel("N") and not B.is_vowel("T")


# ---------------------------------------------------------------------------
# Schema rejection — the HARD CONSTRAINT.
# ---------------------------------------------------------------------------
def test_load_rejects_missing_license(tmp_path):
    import json
    entry = {"id": "x", "file": "wav/x.wav", "syllable": "x",
             "phones": ["EH1"], "source": "say:Fred@pbas70"}  # no license
    (tmp_path / "bank.json").write_text(json.dumps({"entries": [entry]}))
    with pytest.raises(ValueError) as ei:
        B.load_bank(str(tmp_path))
    assert "license" in str(ei.value) and "x" in str(ei.value)


def test_load_rejects_missing_source(tmp_path):
    import json
    entry = {"id": "y", "file": "wav/y.wav", "syllable": "y",
             "phones": ["EH1"], "license": "own-render"}  # no source
    (tmp_path / "bank.json").write_text(json.dumps({"entries": [entry]}))
    with pytest.raises(ValueError) as ei:
        B.load_bank(str(tmp_path))
    assert "source" in str(ei.value) and "y" in str(ei.value)


def test_add_entry_requires_provenance(tmp_path):
    with pytest.raises(ValueError):
        B.add_entry(str(tmp_path), np.zeros(SR // 4, dtype="float32"), SR,
                    {"id": "z", "syllable": "z", "phones": ["EH1"], "source": "say:X"})  # no license


# ---------------------------------------------------------------------------
# add_entry + load round-trip.
# ---------------------------------------------------------------------------
def test_add_entry_roundtrip(tmp_path):
    x = (0.5 * np.sin(2 * np.pi * 220 * np.arange(SR // 2) / SR)).astype("float32")
    e = B.add_entry(str(tmp_path), x, SR, {
        "id": "salt-fred-p70", "syllable": "salt", "phones": ["S", "AO1", "L", "T"],
        "f0_hz": 110.0, "source": "say:Fred@pbas70", "license": "own-render", "attribution": None})
    assert (tmp_path / e["file"]).exists()
    assert e["dur_s"] == pytest.approx(0.5, abs=0.01)

    bank = B.load_bank(str(tmp_path))
    assert len(bank["entries"]) == 1
    got = bank["entries"][0]
    assert got["id"] == "salt-fred-p70"
    assert got["phones"] == ["S", "AO1", "L", "T"]
    assert got["source"] == "say:Fred@pbas70" and got["license"] == "own-render"
    assert got["created"]  # stamped

    # second add appends (manifest grows)
    B.add_entry(str(tmp_path), x, SR, {
        "id": "salt-fred-p110", "syllable": "salt", "phones": ["S", "AO1", "L", "T"],
        "source": "say:Fred@pbas110", "license": "own-render"})
    assert len(B.load_bank(str(tmp_path))["entries"]) == 2


# ---------------------------------------------------------------------------
# Lookup ladder — exact beats nucleus beats vowel-class beats any.
# ---------------------------------------------------------------------------
def _entry(id_, phones, f0=None, dur=0.5):
    return {"id": id_, "file": f"wav/{id_}.wav", "syllable": id_, "phones": phones,
            "f0_hz": f0, "dur_s": dur, "source": "say:Fred@pbas70",
            "license": "own-render", "attribution": None, "created": "2026-07-11"}


def test_lookup_ladder_tiers():
    query = ["SH", "IY1", "N"]  # e.g. the second syllable of "machine"
    bank = {"root": ".", "entries": [
        _entry("exact", ["SH", "IY1", "N"]),      # tier 0: exact
        _entry("nucleus", ["B", "IY1", "T"]),     # tier 1: same stressed nucleus IY (beet)
        _entry("vowelcls", ["Y", "IY0"]),         # tier 2: same vowel class IY, unstressed
        _entry("other", ["K", "AA1", "T"]),       # tier 3: unrelated vowel (cot)
    ]}
    res = B.lookup(bank, query, top=4)
    assert [e["id"] for e, _t in res] == ["exact", "nucleus", "vowelcls", "other"]
    assert [t for _e, t in res] == [0, 1, 2, 3]


def test_lookup_ranks_by_f0_within_tier():
    query = ["K", "AE1", "T"]
    bank = {"root": ".", "entries": [
        _entry("far", ["K", "AE1", "T"], f0=300.0),
        _entry("near", ["K", "AE1", "T"], f0=110.0),
        _entry("nof0", ["K", "AE1", "T"], f0=None),  # no f0 → sinks last
    ]}
    res = B.lookup(bank, query, f0_hint=100.0, top=3)
    assert [e["id"] for e, _ in res] == ["near", "far", "nof0"]


def test_lookup_ranks_by_duration_when_no_f0_hint():
    query = ["K", "AE1", "T"]
    bank = {"root": ".", "entries": [
        _entry("short", ["K", "AE1", "T"], dur=0.2),
        _entry("median", ["K", "AE1", "T"], dur=0.5),
        _entry("long", ["K", "AE1", "T"], dur=0.9),
    ]}
    res = B.lookup(bank, query, top=1)  # median duration is closest to itself
    assert res[0][0]["id"] == "median"


# ---------------------------------------------------------------------------
# Seed-build smoke (skips when `say` is unavailable).
# ---------------------------------------------------------------------------
def test_seed_build_smoke(tmp_path):
    if shutil.which("say") is None:
        pytest.skip("macOS `say` not available")
    from vox_syllabank import seed as SEED
    summary = SEED.build_seed(root=str(tmp_path), wordlist=["salt", "machine", "water"])
    if summary["entries"] == 0:
        pytest.skip("no say voices installed on this host")
    assert summary["entries"] >= 6
    bank = B.load_bank(str(tmp_path))
    assert len(bank["entries"]) == summary["entries"]
    assert all(e.get("license") for e in bank["entries"])
    assert all(e.get("source") for e in bank["entries"])
