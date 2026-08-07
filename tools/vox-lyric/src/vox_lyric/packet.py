"""The lyric packet — the contract between lyric review (this tool) and synthesis
(the phoneme-score tool).

``build_packet(lines, delivery)`` runs the standing review (per-delivery prosody + the
blocklist / choppability gates) and enriches each line with a G2P layer: ARPABET phones
per word, syllable counts, and a flow hint (grid + stress pattern) a score compiler can
build against. ``pronouncing`` (CMUdict) is an OPTIONAL import — when present, phones and
syllable counts are dictionary-backed; when absent (or for out-of-dictionary words) phones
are ``null`` and counts fall back to the vowel-group heuristic.

Phone-format contract (shared across every component):
  - Phones are ARPABET strings WITH stress digits, exactly ``pronouncing.phones_for_word(w)[0]
    .split()`` — e.g. machine -> ["M","AH0","SH","IY1","N"].
  - A vowel nucleus is a phone matching ``[AEIOU].*[0-9]$``.
  - Syllable split: each syllable holds exactly one vowel nucleus; consonants attach forward to
    the next nucleus; trailing consonants with no following vowel append to the last syllable.
    machine -> [["M","AH0"],["SH","IY1","N"]].
  - Words absent from CMUdict: phones ``null``, syllables from the heuristic; consumers handle null.
"""

from __future__ import annotations

import re

from . import prosody

try:  # optional CMUdict-backed G2P
    import pronouncing as _pronouncing
except Exception:  # noqa: BLE001  # pragma: no cover - environment-dependent
    _pronouncing = None

HAS_PRONOUNCING = _pronouncing is not None

PACKET_VERSION = 1

_VOWEL_PHONE = re.compile(r"^[AEIOU].*[0-9]$")


def _is_vowel_phone(phone: str) -> bool:
    """A phone is a vowel nucleus when it starts with a vowel letter and ends in a stress digit."""
    return bool(_VOWEL_PHONE.match(phone))


def phones_for_word(word: str):
    """ARPABET phones (with stress digits) for a word, or ``None`` if unavailable / out-of-dict.

    Exactly ``pronouncing.phones_for_word(w)[0].split()`` per the contract."""
    if _pronouncing is None:
        return None
    clean = re.sub(r"[^a-z']", "", word.lower())
    if not clean:
        return None
    got = _pronouncing.phones_for_word(clean)
    if not got:
        return None
    return got[0].split()


def syllabify(phones):
    """Group phones into syllables per the contract (one vowel nucleus each; consonants attach
    forward; trailing consonants append to the last syllable)."""
    syllables: list[list[str]] = []
    current: list[str] = []
    for p in phones:
        current.append(p)
        if _is_vowel_phone(p):
            syllables.append(current)
            current = []
    if current:  # trailing consonants with no following vowel
        if syllables:
            syllables[-1].extend(current)
        else:
            syllables.append(current)  # degenerate: no vowel at all
    return syllables


def _word_syllable_count(phones, word: str) -> int:
    """Dictionary-backed syllable count when phones are present, else the heuristic."""
    if phones is not None:
        return len(syllabify(phones))
    return prosody.syllables(word)


def _word_stress_chars(phones, syllable_count: int) -> str:
    """Per-syllable stress marks for one word: primary/secondary (1|2) -> ``X``, unstressed (0)
    -> ``x``. Out-of-dictionary words (no phones) default every syllable to unstressed ``x``."""
    if phones is None:
        return "x" * syllable_count
    chars = []
    for p in phones:
        if _is_vowel_phone(p):
            chars.append("X" if p[-1] in ("1", "2") else "x")
    return "".join(chars)


def build_packet(lines, delivery: str, blocklist=None) -> dict:
    """Build the lyric packet for ``lines`` under ``delivery`` (sustained|percussive)."""
    if delivery not in prosody.DELIVERIES:
        raise ValueError(f"unknown delivery {delivery!r} (sustained|percussive)")

    review = prosody.review(lines, delivery, blocklist)  # per-line verdicts + gates

    packet_lines = []
    for row in review["lines"]:
        text = row["line"]
        words_out = []
        stress_parts = []
        syllable_count = 0
        for w in prosody._words(text):
            phones = phones_for_word(w)
            wsyl = _word_syllable_count(phones, w)
            syllable_count += wsyl
            words_out.append({"w": w, "phones": phones, "syllables": int(wsyl)})
            stress_parts.append(_word_stress_chars(phones, wsyl))

        flow_hint = {
            "suggested_grid": 4 if delivery == "percussive" else None,
            "stress_pattern": "".join(stress_parts),
        }
        packet_lines.append({
            "text": text,
            "verdict": row["verdict"],
            "blocklist_flags": row["blocklist_flags"],
            "fit": row["fit"],
            "choppable": row["choppable"],
            "syllable_count": int(syllable_count),
            "words": words_out,
            "flow_hint": flow_hint,
        })

    return {
        "version": PACKET_VERSION,
        "delivery": delivery,
        "generated": None,
        "lines": packet_lines,
        "gates": {
            "n_keep": review["n_keep"],
            "n_rewrite": review["n_rewrite"],
            "blocklist_hits": review["blocklist_hits"],
            "mean_fit": review["mean_fit"],
        },
        "cmudict": HAS_PRONOUNCING,
    }


def validate_packet(d) -> bool:
    """Shape check for a packet dict — returns True when well-formed, False otherwise."""
    try:
        if not isinstance(d, dict):
            return False
        if d.get("version") != PACKET_VERSION:
            return False
        if d.get("delivery") not in prosody.DELIVERIES:
            return False
        if "generated" not in d:
            return False
        if not isinstance(d.get("cmudict"), bool):
            return False

        gates = d.get("gates")
        if not isinstance(gates, dict):
            return False
        for k in ("n_keep", "n_rewrite", "blocklist_hits", "mean_fit"):
            if k not in gates:
                return False

        lines = d.get("lines")
        if not isinstance(lines, list):
            return False
        for ln in lines:
            if not isinstance(ln, dict):
                return False
            for k in ("text", "verdict", "blocklist_flags", "fit",
                      "choppable", "syllable_count", "words", "flow_hint"):
                if k not in ln:
                    return False
            if not isinstance(ln["words"], list):
                return False
            for wd in ln["words"]:
                if not isinstance(wd, dict):
                    return False
                for k in ("w", "phones", "syllables"):
                    if k not in wd:
                        return False
                if wd["phones"] is not None and not isinstance(wd["phones"], list):
                    return False
            fh = ln["flow_hint"]
            if not isinstance(fh, dict):
                return False
            for k in ("suggested_grid", "stress_pattern"):
                if k not in fh:
                    return False
        return True
    except Exception:  # noqa: BLE001
        return False
