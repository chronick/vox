"""The TONGUE score — the one intermediate representation between words and every engine.

A score is plain JSON/YAML-serialisable data (no numpy, no audio) so it round-trips cleanly and
is the durable artifact every synthesis backend consumes:

    score = {
      "meta": {"bpm": float, "key": str|None, "title": str|None},
      "syllables": [
        {"text": str,             # what `say` is handed (a pronounceable syllable fragment)
         "word": str,             # the source word this syllable came from
         "phones": [ARPABET]|None,# ARPABET WITH stress digits (the phone contract); None if OOV
         "start_beat": float,     # onset, in beats from bar start
         "dur_beats": float,      # gated duration, in beats
         "note": str|float|None,  # note name ("A2") OR bare Hz OR None (no pitch imposition)
         "dyn": float,            # 0..1 velocity
         "articulation": "legato"|"spat"|None},
        ...
      ],
    }

`load_score` validates the structure + the phone contract (ARPABET strings with a trailing stress
digit on every vowel). `to_yaml`/`from_yaml` are the on-disk form.
"""

from __future__ import annotations

import re

import yaml

# A vowel nucleus per the phone CONTRACT: starts with a vowel letter, ends with a stress digit,
# e.g. AH0 / IY1 / OW2. Consonant phones (M, SH, N, ...) never match.
_VOWEL_RE = re.compile(r"^[AEIOU].*[0-9]$")
_ARPABET_RE = re.compile(r"^[A-Z]+[0-9]?$")
_ARTICULATIONS = {"legato", "spat", None}


def is_vowel_phone(phone: str) -> bool:
    """True for an ARPABET vowel-with-stress nucleus (the syllable-split anchor)."""
    return bool(_VOWEL_RE.match(phone))


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"invalid TONGUE score: {msg}")


def _validate_phones(phones, where: str) -> None:
    if phones is None:
        return
    _require(isinstance(phones, list), f"{where}.phones must be a list or null")
    for ph in phones:
        _require(isinstance(ph, str) and bool(_ARPABET_RE.match(ph)),
                 f"{where}.phones has a non-ARPABET entry {ph!r}")


def _validate_syllable(syl, i: int) -> dict:
    where = f"syllables[{i}]"
    _require(isinstance(syl, dict), f"{where} must be a mapping")
    _require(isinstance(syl.get("text"), str) and syl["text"] != "", f"{where}.text must be a non-empty str")
    _require(isinstance(syl.get("word"), str), f"{where}.word must be a str")
    _validate_phones(syl.get("phones"), where)
    for k in ("start_beat", "dur_beats"):
        v = syl.get(k)
        _require(isinstance(v, (int, float)) and not isinstance(v, bool), f"{where}.{k} must be a number")
    note = syl.get("note")
    _require(note is None or isinstance(note, (str, int, float)) and not isinstance(note, bool),
             f"{where}.note must be a note-name str, Hz number, or null")
    dyn = syl.get("dyn", 1.0)
    _require(isinstance(dyn, (int, float)) and not isinstance(dyn, bool) and 0.0 <= dyn <= 1.0,
             f"{where}.dyn must be in [0, 1]")
    art = syl.get("articulation")
    _require(art in _ARTICULATIONS, f"{where}.articulation must be legato|spat|null")
    return {
        "text": syl["text"],
        "word": syl["word"],
        "phones": syl.get("phones"),
        "start_beat": float(syl["start_beat"]),
        "dur_beats": float(syl["dur_beats"]),
        "note": note,
        "dyn": float(dyn),
        "articulation": art,
    }


def load_score(path_or_dict) -> dict:
    """Validate + normalise a TONGUE score from a dict, a YAML string, or a file path."""
    if isinstance(path_or_dict, dict):
        raw = path_or_dict
    else:
        raw = from_yaml(path_or_dict)
    _require(isinstance(raw, dict), "top level must be a mapping")
    meta = raw.get("meta", {})
    _require(isinstance(meta, dict), "meta must be a mapping")
    bpm = meta.get("bpm", 120.0)
    _require(isinstance(bpm, (int, float)) and not isinstance(bpm, bool), "meta.bpm must be a number")
    key = meta.get("key")
    _require(key is None or isinstance(key, str), "meta.key must be a str or null")
    title = meta.get("title")
    _require(title is None or isinstance(title, str), "meta.title must be a str or null")
    syls = raw.get("syllables", [])
    _require(isinstance(syls, list), "syllables must be a list")
    return {
        "meta": {"bpm": float(bpm), "key": key, "title": title},
        "syllables": [_validate_syllable(s, i) for i, s in enumerate(syls)],
    }


def to_yaml(score: dict) -> str:
    """Serialise a score to YAML (validated first so the on-disk form is always well-formed)."""
    return yaml.safe_dump(load_score(score), sort_keys=False, allow_unicode=True)


def from_yaml(path_or_text) -> dict:
    """Load a score dict from a YAML string or a path to a ``.yaml`` file. No validation here."""
    text = path_or_text
    try:
        # Treat a short, single-line, existing path as a file; otherwise parse as YAML text.
        import os

        if isinstance(path_or_text, str) and "\n" not in path_or_text and os.path.exists(path_or_text):
            with open(path_or_text, "r", encoding="utf-8") as fh:
                text = fh.read()
    except (OSError, ValueError):
        text = path_or_text
    return yaml.safe_load(text)
