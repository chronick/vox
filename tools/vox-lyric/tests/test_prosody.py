"""Lyric prosody-verifier tests — pure, deterministic.

    cd tools/vox-lyric && uv run pytest -q
"""

from __future__ import annotations

from vox_lyric import prosody


def test_syllable_counts():
    assert prosody.syllables("the") == 1
    assert prosody.syllables("machine") == 2
    assert prosody.syllables("sediment") == 3
    assert prosody.syllables("code") == 1        # silent terminal 'e'
    assert prosody.syllables("") == 0


def test_vowel_vs_plosive_ratio():
    open_line = "aeon our eyes are oceans"
    hard_line = "cut the deck back to black"
    assert prosody.vowel_ratio(open_line) > prosody.vowel_ratio(hard_line)
    assert prosody.plosive_ratio(hard_line) > prosody.plosive_ratio(open_line)


def test_blocklist_filter():
    assert "darkness" in prosody.blocklist_flags("the darkness inside my soul")
    assert "soul" in prosody.blocklist_flags("the darkness inside my soul")
    assert prosody.blocklist_flags("grid-tooth static in the wire") == []


def test_blocklist_is_replaceable():
    custom = frozenset({"wire"})
    assert prosody.blocklist_flags("static in the wire", custom) == ["wire"]
    assert prosody.blocklist_flags("the darkness inside", custom) == []


def test_choppability():
    ok, _ = prosody.choppable("sediment settles over the wire")
    assert ok
    short, reason = prosody.choppable("go")
    assert not short and "too little" in reason


def test_delivery_fit_discriminates():
    sustained = prosody.delivery_fit("a slow tide over open water", "sustained")
    percussive = prosody.delivery_fit("spit the code back, kick the deck", "percussive")
    # each line fits its own delivery better than the other delivery
    assert sustained["fit"] > prosody.delivery_fit("a slow tide over open water", "percussive")["fit"]
    assert percussive["fit"] > prosody.delivery_fit("spit the code back, kick the deck", "sustained")["fit"]


def test_review_flags_blocklisted_as_rewrite():
    r = prosody.review(["the eternal darkness of my broken soul"], "sustained")
    assert r["lines"][0]["verdict"] == "rewrite"
    assert r["n_rewrite"] == 1
    assert set(r["blocklist_hits"]) & {"eternal", "darkness", "broken", "soul"}


def test_review_keeps_clean_line():
    r = prosody.review(["sediment learns the shape of a word"], "sustained")
    assert r["lines"][0]["verdict"] == "keep"
