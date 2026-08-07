"""ARPABET phone inventory, English frequency weights, and pronouncing-backed G2P.

The inventory is the 39-phone CMUdict set WITHOUT stress digits (the unit a voicebank must
cover). Frequency weights are approximate relative occurrence in running English text (schwa AH
and the coronals N/T/S/R dominate; ZH/OY/UH are rare) — used so a coverage score is *weighted*
(missing a rare phone costs less than missing a common one). ``pronouncing`` (CMUdict) is imported
lazily and treated as optional so the module imports even where it is absent; callers check
:func:`have_pronouncing`.
"""

from __future__ import annotations

import re

# The 39 ARPABET phonemes (CMUdict, stress digits stripped). This is the coverage target.
ARPABET_INVENTORY: tuple[str, ...] = (
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
)

# Approximate relative frequency of each phone in running English (normalised to sum ~= 1.0).
# Sourced from published CMUdict/phoneme-frequency tables; exact values are not load-bearing —
# they only tilt the coverage score so a missing schwa hurts more than a missing ZH.
PHONE_FREQUENCY: dict[str, float] = {
    "AH": 0.1174, "N": 0.0700, "T": 0.0700, "S": 0.0505, "R": 0.0496,
    "IH": 0.0494, "L": 0.0424, "D": 0.0411, "IY": 0.0311, "EH": 0.0311,
    "K": 0.0311, "M": 0.0311, "Z": 0.0300, "ER": 0.0290, "P": 0.0290,
    "AE": 0.0271, "W": 0.0203, "B": 0.0203, "AA": 0.0203, "F": 0.0184,
    "EY": 0.0184, "V": 0.0155, "OW": 0.0155, "AO": 0.0126, "NG": 0.0126,
    "HH": 0.0126, "G": 0.0116, "SH": 0.0097, "AY": 0.0097, "Y": 0.0087,
    "JH": 0.0058, "CH": 0.0058, "AW": 0.0048, "TH": 0.0039, "UW": 0.0039,
    "DH": 0.0039, "UH": 0.0019, "OY": 0.0010, "ZH": 0.0005,
}

# A vowel-nucleus phone starts with a vowel letter (in ARPABET, these carry the stress digit).
_VOWEL_START = re.compile(r"^[AEIOU]")
_STRESS = re.compile(r"\d")
_WORD = re.compile(r"[a-z']+")

_PRONOUNCING = None
_TRIED_IMPORT = False


def _pronouncing():
    global _PRONOUNCING, _TRIED_IMPORT
    if not _TRIED_IMPORT:
        _TRIED_IMPORT = True
        try:
            import pronouncing  # type: ignore

            _PRONOUNCING = pronouncing
        except Exception:  # noqa: BLE001 — optional dependency
            _PRONOUNCING = None
    return _PRONOUNCING


def have_pronouncing() -> bool:
    return _pronouncing() is not None


def bare(phone: str) -> str:
    """A phone with any stress digit stripped: ``AH0`` -> ``AH``, ``N`` -> ``N``."""
    return _STRESS.sub("", phone).upper()


def is_vowel(phone: str) -> bool:
    """True when the phone is a vowel nucleus (starts with a vowel letter)."""
    return bool(_VOWEL_START.match(phone.upper()))


def phones_for_word(word: str) -> list[str] | None:
    """ARPABET phones (WITH stress digits) for a word, or ``None`` if out-of-dictionary/absent.

    Exactly ``pronouncing.phones_for_word(w)[0].split()`` per the house contract."""
    p = _pronouncing()
    if p is None:
        return None
    clean = word.lower().strip().strip("'")
    if not clean:
        return None
    prons = p.phones_for_word(clean)
    return prons[0].split() if prons else None


def syllable_count(phones: list[str]) -> int:
    """Number of vowel nuclei in a phone list (== syllable count)."""
    return sum(1 for ph in phones if is_vowel(ph))


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (letters + apostrophes) from free text."""
    return _WORD.findall(text.lower())


def transcript_stats(text: str) -> dict:
    """Phone histogram (stress stripped), syllable count, and word coverage for a text.

    Returns ``{phone_counts, syllables, words, words_in_dict, words_out}``. Out-of-dictionary
    words contribute to ``words`` and ``words_out`` but not to phones/syllables (documented gap;
    they are rare in curated transcripts). ``phone_counts`` maps bare ARPABET -> count.
    """
    counts: dict[str, int] = {}
    syllables = 0
    words = 0
    in_dict = 0
    for tok in tokenize(text):
        words += 1
        phones = phones_for_word(tok)
        if phones is None:
            continue
        in_dict += 1
        syllables += syllable_count(phones)
        for ph in phones:
            b = bare(ph)
            counts[b] = counts.get(b, 0) + 1
    return {
        "phone_counts": counts,
        "syllables": syllables,
        "words": words,
        "words_in_dict": in_dict,
        "words_out": words - in_dict,
    }


def coverage(phone_counts: dict[str, int], rare_threshold: int = 2) -> dict:
    """Phone coverage of an aggregate ``phone_counts`` against the full ARPABET inventory.

    Returns ``present`` / ``missing`` / ``rare`` phone lists, a raw coverage %
    (present / 39) and a frequency-WEIGHTED coverage % (sum of English weights of present phones).
    ``rare`` = present but with count <= ``rare_threshold`` (thin coverage worth flagging).
    """
    present = sorted(p for p in ARPABET_INVENTORY if phone_counts.get(p, 0) > 0)
    missing = [p for p in ARPABET_INVENTORY if phone_counts.get(p, 0) == 0]
    rare = sorted(p for p in present if phone_counts.get(p, 0) <= rare_threshold)

    raw_pct = 100.0 * len(present) / len(ARPABET_INVENTORY)
    total_w = sum(PHONE_FREQUENCY.get(p, 0.0) for p in ARPABET_INVENTORY)
    got_w = sum(PHONE_FREQUENCY.get(p, 0.0) for p in present)
    weighted_pct = 100.0 * got_w / total_w if total_w > 0 else 0.0

    # Order missing by how much they cost (rarest-first list is least useful; sort by weight desc).
    missing_ranked = sorted(missing, key=lambda p: PHONE_FREQUENCY.get(p, 0.0), reverse=True)
    return {
        "present": present,
        "missing": missing_ranked,
        "rare": rare,
        "raw_pct": round(raw_pct, 1),
        "weighted_pct": round(weighted_pct, 1),
        "n_present": len(present),
        "n_inventory": len(ARPABET_INVENTORY),
    }
