"""Grapheme-to-phoneme + syllabification — pronouncing (cmudict) backed, pure logic.

Every consumer of a TONGUE score shares ONE phone contract (see the task spec):

  * Phones are ARPABET strings WITH stress digits, exactly ``pronouncing.phones_for_word(w)[0]``
    split on whitespace — e.g. machine -> ["M","AH0","SH","IY1","N"].
  * A syllable is a phone group with exactly one vowel nucleus (a phone matching ``[AEIOU].*[0-9]$``);
    consonants attach FORWARD to the next nucleus; trailing consonants with no following vowel
    append to the last syllable. machine -> [["M","AH0"],["SH","IY1","N"]].
  * Words not in cmudict: phones fall back to null, and syllable count comes from a letter
    vowel-group heuristic (consumers must handle null phones).

This module is audio-free and deterministic, so it is unit-testable standalone.
"""

from __future__ import annotations

from .schema import is_vowel_phone

_LETTER_VOWELS = set("aeiouy")


def word_phones(word: str):
    """ARPABET phones (with stress digits) for ``word``, or ``None`` if it is not in cmudict."""
    import pronouncing

    prons = pronouncing.phones_for_word(word.lower().strip())
    if not prons:
        return None
    return prons[0].split()


def split_syllables(phones):
    """Group phones into syllables per the CONTRACT (one vowel nucleus each, onsets attach forward).

    Returns a list of phone-groups. ``None``/empty input -> ``[]``.
    """
    if not phones:
        return []
    syllables = []
    buffer = []  # leading consonants waiting to attach forward to the next nucleus
    for ph in phones:
        buffer.append(ph)
        if is_vowel_phone(ph):
            syllables.append(buffer)
            buffer = []
    if buffer:  # trailing consonants with no following vowel -> append to the last syllable
        if syllables:
            syllables[-1].extend(buffer)
        else:
            syllables.append(buffer)  # all-consonant token (no nucleus at all)
    return syllables


def _vowel_groups(word: str):
    """Index ranges ``(start, end)`` of each maximal run of vowel letters in ``word``."""
    w = word.lower()
    groups = []
    i = 0
    while i < len(w):
        if w[i] in _LETTER_VOWELS:
            j = i
            while j < len(w) and w[j] in _LETTER_VOWELS:
                j += 1
            groups.append((i, j))
            i = j
        else:
            i += 1
    return groups


def _letter_syllable_count(word: str) -> int:
    """Heuristic syllable count from letters (number of vowel groups, >= 1)."""
    return max(1, len(_vowel_groups(word)))


def split_word_text(word: str, n: int):
    """Split ``word``'s letters into ``n`` fragments at vowel-group boundaries (onsets attach
    forward), so each fragment stays pronounceable by ``say``: machine, 2 -> ["ma", "chine"].

    Cuts land just AFTER a vowel run (the following consonant cluster becomes the next fragment's
    onset). If the word has fewer vowel groups than ``n``, fall back to a proportional length split.
    """
    if n <= 1:
        return [word]
    groups = _vowel_groups(word)
    if len(groups) >= n:
        cuts = [groups[k][1] for k in range(n - 1)]
    else:
        L = len(word)
        cuts = [round(L * (k + 1) / n) for k in range(n - 1)]
    cuts = sorted({c for c in cuts if 0 < c < len(word)})
    # If dedup/clamping dropped cuts, top up with proportional splits so we still yield n pieces.
    while len(cuts) < n - 1:
        L = len(word)
        proposal = sorted({round(L * (k + 1) / n) for k in range(n - 1)} - set(cuts))
        added = False
        for c in proposal:
            if 0 < c < len(word) and c not in cuts:
                cuts.append(c)
                cuts.sort()
                added = True
                break
        if not added:
            break
    pieces = []
    prev = 0
    for c in cuts:
        pieces.append(word[prev:c])
        prev = c
    pieces.append(word[prev:])
    return [p for p in pieces if p] or [word]


def syllabify_line(line: str):
    """Split a line of text into per-word syllable structure.

    Returns a list of ``{"word", "syllable_texts", "syllable_phones"}`` — one entry per word.
    ``syllable_texts`` is a list of pronounceable letter fragments (one per syllable);
    ``syllable_phones`` is a parallel list of phone-groups, or ``None`` for an out-of-cmudict word.
    """
    out = []
    for raw in line.split():
        word = "".join(ch for ch in raw if ch.isalpha() or ch == "'")
        if not word:
            continue
        phones = word_phones(word)
        if phones is not None:
            groups = split_syllables(phones)
            n = max(1, len(groups))
            texts = split_word_text(word, n)
            # Guard: keep texts and phone-groups the same length (splitter is heuristic).
            if len(texts) != len(groups):
                n = len(groups)
                texts = split_word_text(word, n) if n > 1 else [word]
            out.append({"word": word, "syllable_texts": texts, "syllable_phones": groups})
        else:
            n = _letter_syllable_count(word)
            texts = split_word_text(word, n)
            out.append({"word": word, "syllable_texts": texts, "syllable_phones": None})
    return out
