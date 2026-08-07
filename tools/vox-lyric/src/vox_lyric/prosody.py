"""Text-side prosody analysis for lyric writing — pure python, stdlib only.

Lyrics written for synthesis can be verified like any other spec: per-line prosody
metrics plus two hard gates, run before any audio exists.

Prosody differs by delivery:
  sustained  — vowel density, open syllables, long arcs (melisma room).
  percussive — plosive density, tight syllable budgets (flow room).

Gates (hard, both deliveries):
  blocklist filter — flags on-the-nose vocabulary (evocative beats literal; the default
                     list targets cliché "dark" words and is replaceable per project).
  choppability     — every line must survive being chopped into syllables.

Heuristic, English, v0 — a real G2P (CMUdict) sharpens the syllable + phoneme counts in
the packet layer. The point is a fast, deterministic pre-filter, not a phonetician.
"""

from __future__ import annotations

import re

VOWELS = set("aeiou")
PLOSIVES = set("ptkbdg")
# On-the-nose vocabulary — the default blocklist (evocative, never literal). Replace per
# project via the ``blocklist`` argument / ``--blocklist`` file.
DEFAULT_BLOCKLIST = frozenset({
    "darkness", "shadow", "shadows", "pain", "soul", "souls", "demon", "demons", "eternal",
    "forever", "tears", "blood", "hell", "heaven", "death", "die", "dying", "abyss",
    "suffering", "torment", "nightmare", "despair", "sorrow", "broken", "empty", "void",
})

DELIVERIES = ("sustained", "percussive")


def syllables(word: str) -> int:
    """Vowel-group syllable estimate (English heuristic): count vowel runs, drop a silent final
    'e', floor at 1 for any alphabetic token."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and w[-2] not in "aeiouy":
        n -= 1  # silent terminal 'e'
    return max(n, 1)


def _letters(line: str) -> str:
    return re.sub(r"[^a-z]", "", line.lower())


def _words(line: str) -> list[str]:
    return list(re.findall(r"[a-zA-Z']+", line))


def vowel_ratio(line: str) -> float:
    letters = _letters(line)
    return round(sum(c in VOWELS for c in letters) / len(letters), 4) if letters else 0.0


def plosive_ratio(line: str) -> float:
    cons = [c for c in _letters(line) if c not in VOWELS]
    return round(sum(c in PLOSIVES for c in cons) / len(cons), 4) if cons else 0.0


def open_syllable_frac(line: str) -> float:
    """Fraction of words ending in a vowel sound (open syllables — melisma-friendly)."""
    words = _words(line)
    if not words:
        return 0.0
    return round(sum(1 for w in words if re.sub(r"[^a-z]", "", w.lower())[-1:] in VOWELS) / len(words), 4)


def line_syllables(line: str) -> int:
    return sum(syllables(w) for w in _words(line))


def blocklist_flags(line: str, blocklist=None) -> list[str]:
    bl = DEFAULT_BLOCKLIST if blocklist is None else blocklist
    return sorted({w.lower() for w in _words(line) if w.lower() in bl})


def choppable(line: str) -> tuple[bool, str]:
    """A line survives being chopped into syllables when it has enough syllabic content to slice
    and no single unwieldy token. < 3 total syllables = too little to chop; any word > 6 syllables
    = a slice hazard."""
    words = _words(line)
    total = line_syllables(line)
    if total < 3:
        return False, f"only {total} syllables — too little to chop"
    longest = max((syllables(w) for w in words), default=0)
    if longest > 6:
        return False, f"a {longest}-syllable word resists clean chopping"
    return True, "ok"


def delivery_fit(line: str, delivery: str) -> dict:
    """Score a line's fit to its delivery's prosody (0–1) + the failing constraint if weak."""
    vr, pr = vowel_ratio(line), plosive_ratio(line)
    osf = open_syllable_frac(line)
    syl = line_syllables(line)
    if delivery == "sustained":
        fit = 0.5 * min(vr / 0.45, 1.0) + 0.3 * osf + 0.2 * min(syl / 8.0, 1.0)
        weak = "low vowel density / few open syllables for a sustained arc" if fit < 0.5 else None
    elif delivery == "percussive":
        fit = 0.6 * min(pr / 0.35, 1.0) + 0.4 * (1.0 - min(syl / 12.0, 1.0))
        weak = "low plosive punch for a percussive delivery" if fit < 0.5 else None
    else:
        raise ValueError(f"unknown delivery {delivery!r} (sustained|percussive)")
    return {"fit": round(min(fit, 1.0), 3), "weak": weak, "vowel_ratio": vr,
            "plosive_ratio": pr, "open_syllable_frac": osf, "syllables": syl}


def review_line(line: str, delivery: str, blocklist=None) -> dict:
    flags = blocklist_flags(line, blocklist)
    chops, reason = choppable(line)
    fit = delivery_fit(line, delivery)
    verdict = "keep"
    if flags or not chops or fit["fit"] < 0.4:
        verdict = "rewrite"
    return {"line": line, "delivery": delivery, "verdict": verdict, "blocklist_flags": flags,
            "choppable": chops, "chop_reason": reason, **fit}


def review(lines, delivery: str, blocklist=None) -> dict:
    rows = [review_line(ln, delivery, blocklist) for ln in lines if ln.strip()]
    n = len(rows)
    return {
        "delivery": delivery,
        "lines": rows,
        "n_lines": n,
        "n_keep": sum(1 for r in rows if r["verdict"] == "keep"),
        "n_rewrite": sum(1 for r in rows if r["verdict"] == "rewrite"),
        "blocklist_hits": sorted({c for r in rows for c in r["blocklist_flags"]}),
        "mean_fit": round(sum(r["fit"] for r in rows) / n, 3) if n else 0.0,
    }
