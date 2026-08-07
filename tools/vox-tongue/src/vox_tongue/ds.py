"""ds — emit a DiffSinger ``.ds`` segment from a TONGUE score.

DiffSinger is the closest living relative of the TONGUE score: its acoustic model consumes exactly
*phoneme sequence + per-phoneme duration + F0 curve* — the three things a TONGUE syllable already
carries (phones, beat-gated durations, per-syllable notes). This module is the ``TONGUE -> .ds``
bridge. It is **audio-free and dependency-light** (pure Python + ``json``): note→Hz is reimplemented
locally so emitting a ``.ds`` never drags in ``pyworld``/``numpy``.

The ``.ds`` schema (openvpi / OpenUtau English banks) stores space-separated strings:

    ph_seq      phones (our TONGUE phones, ARPABET stress-stripped + lowercased, + SP silences)
    ph_dur      per-phoneme durations, in SECONDS
    ph_num      phones-per-note (groups ph_seq under each note_seq entry)
    note_seq    per-note names ("A2", "C#4", "rest"); one note per TONGUE syllable, rests for SP
    note_dur    per-note durations, in seconds (= sum of that note's ph_dur)
    note_slur   per-note slur flags (all "0" — TONGUE is one-note-per-syllable, no melisma yet)
    f0_seq      explicit Hz curve, flat per note, sampled at ``f0_timestep`` (optional)
    f0_timestep the f0 hop, in seconds

A ``.ds`` file on disk is a JSON **array** of segments; ``to_ds`` returns ONE segment (a score = one
phrase). Word-boundary ``SP`` silence tokens are emitted between words (and wherever the score leaves
a rest); the leading ``SP`` carries any pre-roll and a zero-duration trailing ``SP`` marks phrase end.

### ARPABET -> DiffSinger English phoneme map

English DiffSinger voicebanks use an ARPABET-derived inventory ("DIFFS EN X" = ARPABET + ``[ax]`` +
``[dx]``), so the TONGUE phones map ~1:1 — **strip the stress digit, lowercase**. cmudict never emits
``ax``/``dx`` (schwa always comes through as ``AH0`` → ``ah``), so ``ARPABET_TO_DS`` covers the whole
cmudict phone inventory (the 15 vowels + 24 consonants of CMU ARPABET). Unmapped phones raise.
"""

from __future__ import annotations

import math
import re

from .schema import is_vowel_phone, load_score

# --- ARPABET (stress-stripped) -> DiffSinger English phoneme -----------------------------------
# The 39 CMU ARPABET phones. DiffSinger English banks lowercase them; ax/dx are bank extras cmudict
# never produces, so this table is complete coverage of everything ``pronouncing`` can emit.
ARPABET_TO_DS = {
    # 15 vowels
    "AA": "aa", "AE": "ae", "AH": "ah", "AO": "ao", "AW": "aw", "AY": "ay",
    "EH": "eh", "ER": "er", "EY": "ey", "IH": "ih", "IY": "iy",
    "OW": "ow", "OY": "oy", "UH": "uh", "UW": "uw",
    # 24 consonants
    "B": "b", "CH": "ch", "D": "d", "DH": "dh", "F": "f", "G": "g", "HH": "hh",
    "JH": "jh", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ng", "P": "p",
    "R": "r", "S": "s", "SH": "sh", "T": "t", "TH": "th", "V": "v", "W": "w",
    "Y": "y", "Z": "z", "ZH": "zh",
}

SP = "SP"  # DiffSinger silence / word-boundary token

_STRESS_RE = re.compile(r"\d+$")
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
_BASE_SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def strip_stress(phone: str) -> str:
    """``"IY1"`` -> ``"IY"`` (consonants pass through unchanged)."""
    return _STRESS_RE.sub("", phone)


def map_phone(phone: str) -> str:
    """ARPABET phone (with or without stress digit) -> DiffSinger English phoneme. Raises on unmapped."""
    base = strip_stress(phone)
    try:
        return ARPABET_TO_DS[base]
    except KeyError as exc:
        raise ValueError(
            f"unmapped ARPABET phone {phone!r} (stripped {base!r}) — not in the DiffSinger English "
            f"phoneme set; extend vox_tongue.ds.ARPABET_TO_DS") from exc


def note_to_hz(name: str) -> float:
    """``"A2"`` / ``"C#4"`` / ``"Eb2"`` -> Hz (A4 = 440, MIDI-standard). Matches vox_larynx.world."""
    m = _NOTE_RE.match(name.strip())
    if not m:
        raise ValueError(f"bad note name {name!r}")
    letter, acc, octv = m.group(1).upper(), m.group(2), int(m.group(3))
    semi = _BASE_SEMI[letter] + (1 if acc == "#" else -1 if acc == "b" else 0)
    midi = (octv + 1) * 12 + semi
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def hz_to_note_name(hz: float) -> str:
    """Hz -> nearest equal-tempered note name in sharp spelling (``0`` / negative -> ``"rest"``)."""
    if hz <= 0:
        return "rest"
    midi = int(round(69 + 12 * math.log2(hz / 440.0)))
    return f"{_SHARP_NAMES[midi % 12]}{midi // 12 - 1}"


def _note_name_and_hz(note):
    """A TONGUE ``note`` (name str | bare Hz | None) -> ``(ds_note_name, exact_hz)``.

    Names round-trip through Hz to a canonical sharp spelling ("Eb2" -> "D#2"); a bare Hz keeps its
    exact value for ``f0_seq`` while ``note_seq`` gets the nearest name; ``None`` -> ``("rest", 0.0)``.
    """
    if note is None:
        return "rest", 0.0
    if isinstance(note, str):
        hz = note_to_hz(note)
        return hz_to_note_name(hz), hz
    hz = float(note)
    return hz_to_note_name(hz), hz


def _distribute(total: float, phones, cons_sec: float):
    """Split a syllable's ``total`` seconds across its phones, consonant-aware.

    Consonants each get a nominal ``cons_sec`` (capped so all consonants never exceed 60% of the
    syllable — long sustained vowels must not crush the consonants, per the research); the vowel(s)
    absorb the remainder. All-consonant fragments split evenly. Sums back to ``total`` exactly.
    """
    n = len(phones)
    if n == 0:
        return []
    if total <= 0:
        return [0.0] * n
    if n == 1:
        return [round(total, 6)]
    is_v = [is_vowel_phone(p) for p in phones]
    n_v = sum(is_v)
    n_c = n - n_v
    if n_v == 0:
        durs = [total / n] * n
    else:
        cons_total = min(n_c * cons_sec, total * 0.6) if n_c else 0.0
        per_c = cons_total / n_c if n_c else 0.0
        per_v = (total - cons_total) / n_v
        durs = [per_v if v else per_c for v in is_v]
    durs = [round(d, 6) for d in durs]
    resid = round(total - sum(durs), 6)  # absorb rounding drift into the last phone
    durs[-1] = round(durs[-1] + resid, 6)
    return durs


def _fmt(x: float) -> str:
    """Compact fixed-point (no numpy scalars, no sci-notation): ``0.06`` not ``6e-2``/``np.float64``."""
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return s or "0"


def _fmt_f0(x: float) -> str:
    s = f"{float(x):.3f}".rstrip("0").rstrip(".")
    return s or "0"


def to_ds(score, *, with_f0: bool = True, f0_timestep: float = 0.024, cons_sec: float = 0.06) -> dict:
    """A validated TONGUE ``score`` -> one DiffSinger ``.ds`` segment dict (all values JSON-clean).

    ``with_f0`` emits a flat-per-note ``f0_seq`` at ``f0_timestep`` (rests -> 0 Hz / unvoiced).
    ``cons_sec`` is the nominal consonant length used by the consonant-aware duration split.
    Raises ``ValueError`` on an OOV syllable (``phones is None``) or an unmapped phone.
    """
    score = load_score(score)
    bpm = score["meta"]["bpm"]
    spb = 60.0 / bpm
    syls = sorted(score["syllables"], key=lambda s: s["start_beat"])
    if not syls:
        raise ValueError("empty score: no syllables to emit as .ds")

    ph_seq, ph_dur, ph_num = [], [], []
    note_seq, note_dur, note_hz, note_slur = [], [], [], []
    words = []

    first_start = syls[0]["start_beat"] * spb
    # Leading SP carries any pre-roll before the first onset (0 if the score starts at beat 0).
    ph_seq.append(SP); ph_dur.append(max(first_start, 0.0)); ph_num.append(1)
    note_seq.append("rest"); note_dur.append(max(first_start, 0.0)); note_hz.append(0.0); note_slur.append(0)

    prev_end = first_start
    prev_word = None
    for i, s in enumerate(syls):
        start = s["start_beat"] * spb
        dur = s["dur_beats"] * spb
        if i > 0:  # SP between words (or wherever the score leaves a rest)
            gap = start - prev_end
            if gap > 1e-9 or s["word"] != prev_word:
                g = max(gap, 0.0)
                ph_seq.append(SP); ph_dur.append(g); ph_num.append(1)
                note_seq.append("rest"); note_dur.append(g); note_hz.append(0.0); note_slur.append(0)
        phones = s["phones"]
        if not phones:
            raise ValueError(
                f"syllable {s['text']!r} (word {s['word']!r}) has no phones — cannot emit .ds "
                "(OOV word: add a pronunciation or drop the syllable before emitting)")
        ds_ph = [map_phone(p) for p in phones]
        durs = _distribute(dur, phones, cons_sec)
        ph_seq.extend(ds_ph); ph_dur.extend(durs); ph_num.append(len(ds_ph))
        name, hz = _note_name_and_hz(s["note"])
        note_seq.append(name); note_dur.append(dur); note_hz.append(hz); note_slur.append(0)
        words.append(s["text"])
        prev_end = max(prev_end, start + dur)
        prev_word = s["word"]

    # Zero-duration trailing SP marks phrase end (DiffSinger convention).
    ph_seq.append(SP); ph_dur.append(0.0); ph_num.append(1)
    note_seq.append("rest"); note_dur.append(0.0); note_hz.append(0.0); note_slur.append(0)

    seg = {
        "offset": 0.0,
        "text": " ".join(words),
        "ph_seq": " ".join(ph_seq),
        "ph_dur": " ".join(_fmt(d) for d in ph_dur),
        "ph_num": " ".join(str(n) for n in ph_num),
        "note_seq": " ".join(note_seq),
        "note_dur": " ".join(_fmt(d) for d in note_dur),
        "note_slur": " ".join(str(x) for x in note_slur),
    }

    if with_f0:
        total = sum(ph_dur)
        n_frames = max(1, int(round(total / f0_timestep)))
        f0 = [0.0] * n_frames
        t = 0.0
        for hz, nd in zip(note_hz, note_dur):
            f_start = int(round(t / f0_timestep))
            f_end = int(round((t + nd) / f0_timestep))
            for k in range(max(0, f_start), min(n_frames, f_end)):
                f0[k] = hz
            t += nd
        seg["f0_seq"] = " ".join(_fmt_f0(v) for v in f0)
        seg["f0_timestep"] = _fmt(f0_timestep)

    return seg
