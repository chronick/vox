"""The syllable bank — schema, license-gated loader, and the lookup fallback ladder.

The bank is a directory (``VOX_BANK_DIR`` / ``~/.vox/bank`` by default) holding a
``bank.json`` manifest plus the referenced WAVs. Each manifest row is an *entry*:

    {id, file, syllable, phones, f0_hz, dur_s, source, license, attribution, created}

HARD CONSTRAINT (the whole point of v0): every entry MUST carry ``source`` + ``license``.
:func:`load_bank` re-validates on read and raises ``ValueError`` naming the offending entry —
so an un-provenanced sample can never silently enter the concat substrate.

The phone contract (shared by every component of the toolchain) lives here as the pair
:func:`syllabify` / :func:`is_vowel`: ARPABET-with-stress strings exactly as
``pronouncing.phones_for_word(w)[0].split()`` returns them, grouped one-vowel-nucleus-per-
syllable with consonants attaching FORWARD to the next nucleus.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from pathlib import Path

# Bumped on any schema/behaviour change (mirrors the op_version convention across the suite).
BANK_SCHEMA_VERSION = "syllabank@1"

# Required keys on every entry. ``phones`` MUST be present but its VALUE may be null
# (words absent from cmudict — consumers handle null); the rest must be present AND non-empty.
REQUIRED_KEYS = ("id", "file", "syllable", "phones", "source", "license")
_NONEMPTY_KEYS = ("id", "file", "syllable", "source", "license")

# A vowel nucleus: an ARPABET phone starting with a vowel letter and ending in a stress digit
# (e.g. AH0, IY1, AE2). Matches the shared toolchain contract verbatim.
_VOWEL_RE = re.compile(r"[AEIOU].*[0-9]$")


# ---------------------------------------------------------------------------
# Phone contract.
# ---------------------------------------------------------------------------
def is_vowel(phone: str) -> bool:
    """True if ``phone`` is a vowel nucleus (``[AEIOU].*[0-9]$``)."""
    return bool(_VOWEL_RE.match(phone))


def syllabify(phones: list[str]) -> list[list[str]]:
    """Group ARPABET phones into syllables: one vowel nucleus each, consonants forward.

    ``["M","AH0","SH","IY1","N"]`` -> ``[["M","AH0"],["SH","IY1","N"]]``. A pending run of
    consonants attaches to the NEXT nucleus; a trailing consonant run with no following vowel
    appends to the last syllable (or, for an all-consonant input, forms a lone group).
    """
    syllables: list[list[str]] = []
    pending: list[str] = []
    for p in phones:
        pending.append(p)
        if is_vowel(p):
            syllables.append(pending)
            pending = []
    if pending:  # trailing consonants with no following vowel
        if syllables:
            syllables[-1].extend(pending)
        else:
            syllables.append(pending)
    return syllables


def word_syllables(word: str):
    """``(phones_or_None, syllables)`` for a word.

    Uses cmudict via ``pronouncing``. Words not in the dictionary fall back to a vowel-group
    heuristic (count orthographic vowel runs) and set ``phones`` to ``None`` — consumers must
    handle the null. ``syllables`` is a list of phone-lists (empty phone-lists when null).
    """
    import pronouncing

    prons = pronouncing.phones_for_word(word)
    if prons:
        phones = prons[0].split()
        return phones, syllabify(phones)
    # Not in cmudict: heuristic syllable count from orthographic vowel runs; phones null.
    n = max(1, len(re.findall(r"[aeiouy]+", word.lower())))
    return None, [[] for _ in range(n)]


# ---------------------------------------------------------------------------
# Bank root + manifest I/O.
# ---------------------------------------------------------------------------
def bank_root(root: str | None = None) -> Path:
    """Resolve the bank root: explicit arg > ``VOX_BANK_DIR`` > ``~/.vox/bank``."""
    if root:
        return Path(root).expanduser()
    return Path(os.environ.get("VOX_BANK_DIR", "~/.vox/bank")).expanduser()


def _manifest_path(root: Path) -> Path:
    return root / "bank.json"


def _validate_entry(entry: dict, where: str) -> None:
    """Raise ``ValueError`` (naming the entry) if any required key is missing/empty."""
    for key in REQUIRED_KEYS:
        if key not in entry:
            raise ValueError(f"bank entry {where}: missing required field {key!r}")
    for key in _NONEMPTY_KEYS:
        val = entry.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            raise ValueError(f"bank entry {where}: empty required field {key!r}")
    # Attribution-required licenses (review fix): CC-BY-family entries MUST carry a
    # non-empty attribution — presence of `license` alone is not licence compliance.
    lic = str(entry.get("license", "")).strip().lower().replace("_", "-").replace(" ", "-")
    if lic.startswith(("cc-by", "ccby")):
        attr = entry.get("attribution")
        if attr is None or not str(attr).strip():
            raise ValueError(
                f"bank entry {where}: license {entry.get('license')!r} requires a "
                f"non-empty 'attribution' field")


def load_bank(root: str | None = None) -> dict:
    """Load + VALIDATE the bank manifest. Raises ``ValueError`` naming any bad entry.

    Returns ``{"root": <str>, "entries": [entry, ...]}``. A missing manifest is an empty bank.
    """
    r = bank_root(root)
    mf = _manifest_path(r)
    if not mf.exists():
        return {"root": str(r), "entries": []}
    data = json.loads(mf.read_text())
    entries = data.get("entries", [])
    for i, entry in enumerate(entries):
        where = entry.get("id") or f"#{i}"
        _validate_entry(entry, where)
    return {"root": str(r), "entries": entries}


def save_bank(bank: dict, root: str | None = None) -> Path:
    """Write the manifest (validating every entry first). Returns the manifest path."""
    r = bank_root(root or bank.get("root"))
    r.mkdir(parents=True, exist_ok=True)
    entries = bank.get("entries", [])
    for i, entry in enumerate(entries):
        _validate_entry(entry, entry.get("id") or f"#{i}")
    mf = _manifest_path(r)
    mf.write_text(json.dumps({"schema": BANK_SCHEMA_VERSION, "entries": entries}, indent=2))
    return mf


def _safe_stem(entry_id: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", entry_id.lower()).strip("_") or "entry"


def add_entry(root: str | None, samples, sr: int, meta: dict) -> dict:
    """Write ``samples`` as a WAV under the bank root and append a validated manifest row.

    ``meta`` supplies the schema fields (id/syllable/phones/source/license, and optionally
    f0_hz/attribution). ``file``/``dur_s``/``created`` are computed here. Returns the entry.
    Raises ``ValueError`` (via validation) if provenance is missing — the row is never written.
    """
    import datetime as _dt

    import numpy as np
    import soundfile as sf

    r = bank_root(root)
    (r / "wav").mkdir(parents=True, exist_ok=True)

    arr = np.ascontiguousarray(np.asarray(samples, dtype="float32"))
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    dur_s = float(len(arr) / sr) if sr else 0.0

    entry_id = meta.get("id")
    if not entry_id:
        raise ValueError("add_entry: meta must include a non-empty 'id'")
    rel = f"wav/{_safe_stem(entry_id)}.wav"

    f0 = meta.get("f0_hz")
    entry = {
        "id": str(entry_id),
        "file": rel,
        "syllable": meta.get("syllable"),
        "phones": meta.get("phones"),
        "f0_hz": (None if f0 is None else float(f0)),
        "dur_s": round(dur_s, 4),
        "source": meta.get("source"),
        "license": meta.get("license"),
        "attribution": meta.get("attribution"),
        "created": meta.get("created") or _dt.datetime.now().isoformat(timespec="seconds"),
    }
    # Validate BEFORE touching disk so a provenance-less entry can't leave a stray WAV.
    _validate_entry(entry, entry["id"])

    sf.write(str(r / rel), arr, sr, subtype="PCM_16")

    bank = load_bank(str(r))
    bank["entries"].append(entry)
    save_bank(bank, str(r))
    return entry


# ---------------------------------------------------------------------------
# Lookup fallback ladder.
# ---------------------------------------------------------------------------
def _vowel_phones(phones) -> list[str]:
    return [p for p in (phones or []) if is_vowel(p)]


def _vowel_class(phone: str) -> str:
    """``IY1`` -> ``IY`` (drop the trailing stress digit)."""
    return re.sub(r"[0-9]$", "", phone)


def _stressed_vowel_class(phones) -> str | None:
    """Vowel class of the stressed nucleus (first vowel carrying stress 1 or 2), else None."""
    for p in _vowel_phones(phones):
        if p[-1] in "12":
            return _vowel_class(p)
    return None


def _primary_vowel_class(phones) -> str | None:
    """Vowel class ignoring stress: prefer a stressed nucleus, else the first vowel."""
    vowels = _vowel_phones(phones)
    if not vowels:
        return None
    for p in vowels:
        if p[-1] in "12":
            return _vowel_class(p)
    return _vowel_class(vowels[0])


def _tier_of(query_phones, entry_phones) -> int:
    """Match tier for an entry against the query (0=exact .. 3=any). Lower is better."""
    if entry_phones is not None and query_phones is not None and list(entry_phones) == list(query_phones):
        return 0
    q_stress, e_stress = _stressed_vowel_class(query_phones), _stressed_vowel_class(entry_phones)
    if q_stress is not None and q_stress == e_stress:
        return 1
    q_vowel, e_vowel = _primary_vowel_class(query_phones), _primary_vowel_class(entry_phones)
    if q_vowel is not None and q_vowel == e_vowel:
        return 2
    return 3


def _rank_key(f0_hint: float | None, median_dur: float):
    if f0_hint is not None:
        # Entries with no measured f0 sink to the bottom of the tier (infinite distance).
        def key(e):
            f0 = e.get("f0_hz")
            return abs(f0 - f0_hint) if f0 is not None else float("inf")
    else:
        def key(e):
            return abs((e.get("dur_s") or 0.0) - median_dur)
    return key


def lookup(bank: dict, phones, f0_hint: float | None = None, top: int = 5):
    """Rank bank entries against ``phones`` through the fallback ladder.

    Tiers: (0) exact phone-sequence match; (1) same stressed vowel nucleus; (2) same vowel
    ignoring stress; (3) any entry. Within a tier, rank by f0 distance to ``f0_hint`` when
    given, else by duration closeness to the tier's median duration. Returns a list of
    ``(entry, tier)`` of length up to ``top``, best first.
    """
    entries = bank.get("entries", [])
    buckets: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
    for e in entries:
        buckets[_tier_of(phones, e.get("phones"))].append(e)

    results: list[tuple[dict, int]] = []
    for tier in (0, 1, 2, 3):
        cands = buckets[tier]
        if not cands:
            continue
        durs = [e.get("dur_s") or 0.0 for e in cands]
        median_dur = statistics.median(durs) if durs else 0.0
        for e in sorted(cands, key=_rank_key(f0_hint, median_dur)):
            results.append((e, tier))
            if len(results) >= top:
                return results
    return results
