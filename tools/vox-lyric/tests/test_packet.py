"""Lyric-packet tests — the review→score-compiler seam.

    cd tools/vox-lyric && uv run pytest -q
"""

from __future__ import annotations

import copy

from vox_lyric import packet


def test_phones_present_with_stress_digits():
    pk = packet.build_packet(["machine gun static"], "percussive")
    assert pk["version"] == 1
    assert pk["delivery"] == "percussive"
    assert pk["generated"] is None
    machine = pk["lines"][0]["words"][0]
    assert machine["w"] == "machine"
    # exactly pronouncing.phones_for_word("machine")[0].split()
    assert machine["phones"] == ["M", "AH0", "SH", "IY1", "N"]
    assert machine["syllables"] == 2
    # every phone list carries stress digits on its vowel nuclei
    assert any(p.endswith(("0", "1", "2")) for p in machine["phones"])


def test_stress_pattern_and_syllable_count_for_known_line():
    pk = packet.build_packet(["machine gun static"], "percussive")
    ln = pk["lines"][0]
    # machine -> xX, gun -> X, static -> Xx  ==>  xXXXx
    assert ln["flow_hint"]["stress_pattern"] == "xXXXx"
    assert ln["syllable_count"] == 5
    # percussive flow hint suggests a 4-grid
    assert ln["flow_hint"]["suggested_grid"] == 4


def test_sustained_flow_hint_has_null_grid():
    pk = packet.build_packet(["a slow tide over open water"], "sustained")
    assert pk["lines"][0]["flow_hint"]["suggested_grid"] is None


def test_blocklisted_line_carries_rewrite_verdict_and_flags():
    pk = packet.build_packet(["the eternal darkness of my broken soul"], "sustained")
    ln = pk["lines"][0]
    assert ln["verdict"] == "rewrite"
    assert set(ln["blocklist_flags"]) & {"eternal", "darkness", "broken", "soul"}
    assert pk["gates"]["n_rewrite"] == 1
    assert set(pk["gates"]["blocklist_hits"]) & {"eternal", "darkness", "broken", "soul"}


def test_unknown_word_phones_null_handled():
    pk = packet.build_packet(["static grglthwump wire"], "sustained")
    words = {wd["w"]: wd for wd in pk["lines"][0]["words"]}
    assert words["grglthwump"]["phones"] is None            # out-of-dictionary -> null
    assert words["grglthwump"]["syllables"] >= 1            # heuristic fallback, floored at 1
    assert words["static"]["phones"] is not None            # in-dictionary neighbours still resolve
    # stress string still spans every syllable in the line (unknown syllables default to 'x')
    total_syl = sum(wd["syllables"] for wd in pk["lines"][0]["words"])
    assert len(pk["lines"][0]["flow_hint"]["stress_pattern"]) == total_syl


def test_validate_packet_passes_and_fails_on_mutilation():
    pk = packet.build_packet(["machine gun static", "grid tooth static in the wire"], "percussive")
    assert packet.validate_packet(pk) is True

    missing_gates = copy.deepcopy(pk)
    del missing_gates["gates"]
    assert packet.validate_packet(missing_gates) is False

    bad_word = copy.deepcopy(pk)
    del bad_word["lines"][0]["words"][0]["phones"]
    assert packet.validate_packet(bad_word) is False

    wrong_version = copy.deepcopy(pk)
    wrong_version["version"] = 2
    assert packet.validate_packet(wrong_version) is False


def test_cmudict_flag_reflects_pronouncing_availability():
    pk = packet.build_packet(["machine"], "percussive")
    assert pk["cmudict"] is packet.HAS_PRONOUNCING
