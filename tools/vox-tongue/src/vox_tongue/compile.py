"""compile — words + a melody (+ optional FLOW cadence) -> a validated TONGUE score.

The compiler is audio-free and deterministic. Placement is either:
  * FLOW-driven (``flow_pattern`` given): import ``vox_flow.flow.compile_flow`` (sibling tool on
    PYTHONPATH) and place one syllable per grid onset — start/dur come straight from the onset
    events (converted seconds -> beats), so placement matches ``compile_flow`` exactly; or
  * even quarter-note spread (default): syllable ``i`` starts at beat ``i`` for one beat.

Notes cycle through ``melody`` (note names like ``"A2"`` or bare Hz). ``note_to_hz`` lives in the
sibling ``vox_larynx`` tool and is resolved lazily at render time — the score stores notes as
authored so it stays a pure data artifact.
"""

from __future__ import annotations

from .g2p import split_syllables, split_word_text, syllabify_line
from .schema import load_score


def _flatten_syllables(lines):
    """Flatten lines -> ordered list of ``{text, word, phones}`` syllable atoms."""
    atoms = []
    for line in lines:
        for w in syllabify_line(line):
            texts = w["syllable_texts"]
            groups = w["syllable_phones"]
            for i, text in enumerate(texts):
                if groups is None:
                    phones = None
                elif i < len(groups):
                    phones = list(groups[i])
                else:
                    phones = None
                atoms.append({"text": text, "word": w["word"], "phones": phones})
    return atoms


def _syl(atom, note, *, start_beat, dur_beats, dyn=1.0, articulation=None):
    return {
        "text": atom["text"],
        "word": atom["word"],
        "phones": atom["phones"],
        "start_beat": float(start_beat),
        "dur_beats": float(dur_beats),
        "note": note,
        "dyn": float(dyn),
        "articulation": articulation,
    }


def compile(lines, melody, bpm=120, flow_pattern=None, delivery=None):
    """Compile lines + a cycled melody into a TONGUE score.

    ``lines`` — list of text lines. ``melody`` — list of note names or Hz, cycled across syllables.
    ``flow_pattern`` — an optional FLOW grid pattern for authored placement. ``delivery`` —
    reserved hook; accepted for signature stability (documented no-op today).
    Returns a validated score dict.
    """
    if isinstance(lines, str):
        lines = [lines]
    melody = list(melody)
    if not melody:
        raise ValueError("melody must have at least one note")
    atoms = _flatten_syllables(lines)

    syllables = []
    if flow_pattern:
        from vox_flow.flow import compile_flow  # sibling tool, resolved on PYTHONPATH

        flow_score = compile_flow(flow_pattern, bpm=bpm,
                                  syllables=[a["text"] for a in atoms] or None)
        # Lyric-integrity gate: a FLOW pattern whose onset count disagrees with the
        # syllable count silently cycles/drops syllables — hard-error instead of mangling words.
        n_onsets = len(flow_score["events"])
        if atoms and n_onsets != len(atoms):
            raise ValueError(
                f"compile: FLOW pattern has {n_onsets} onsets but {len(atoms)} syllables "
                f"(pattern {flow_pattern!r}) — placement would drop/cycle syllables")
        spb = 60.0 / float(bpm)  # seconds per beat
        for i, ev in enumerate(flow_score["events"]):
            atom = atoms[i % len(atoms)] if atoms else {"text": ev.get("syllable") or "la",
                                                        "word": "", "phones": None}
            note = melody[i % len(melody)]
            dyn = 1.0 if ev.get("accent") else 0.62
            syllables.append(_syl(atom, note, start_beat=ev["t"] / spb,
                                  dur_beats=ev["dur"] / spb, dyn=dyn,
                                  articulation="spat"))
    else:
        for i, atom in enumerate(atoms):
            note = melody[i % len(melody)]
            syllables.append(_syl(atom, note, start_beat=float(i), dur_beats=1.0))

    meta = {"bpm": float(bpm), "key": None, "title": None}
    if delivery is not None:
        meta["key"] = None  # delivery is reserved; keep meta schema-clean (documented no-op)
    return load_score({"meta": meta, "syllables": syllables})


# ---------------------------------------------------------------------------
# compile_packet — compile a vox-lyric packet (words + gates) into a score.
# ---------------------------------------------------------------------------
def _packet_atoms(line: dict):
    """Flatten one packet line's ``words`` into syllable atoms, taking phones FROM the packet
    (never re-running G2P). Per-word phones are grouped with the shared syllabify contract;
    text fragments are orthographic (``say`` fallback shaping only)."""
    atoms = []
    for wd in line.get("words", []):
        w = wd.get("w", "")
        phones = wd.get("phones")
        nsyl = max(1, int(wd.get("syllables", 1)))
        if phones is not None:
            groups = split_syllables(phones) or [None] * nsyl
        else:
            groups = [None] * nsyl
        n = len(groups)
        texts = split_word_text(w, n)
        if len(texts) != n:  # keep texts and phone-groups the same length (splitter is heuristic)
            texts = (texts + [texts[-1]] * n)[:n] if texts else [w] * n
        for i in range(n):
            grp = groups[i]
            atoms.append({"text": texts[i], "word": w,
                          "phones": (list(grp) if grp else None)})
    return atoms


def _stress_line_pattern(line: dict) -> str:
    """Derive a FLOW pattern from a line's ``flow_hint``: map the per-syllable stress string
    (X = stressed onset, x = unstressed onset) onto the grid, with a rest between words."""
    stress = line.get("flow_hint", {}).get("stress_pattern") or ""
    chunks = []
    idx = 0
    for wd in line.get("words", []):
        n = max(1, int(wd.get("syllables", 1)))
        chunk = stress[idx:idx + n]
        if len(chunk) != n:  # pad short/empty stress runs to one onset per syllable
            chunk = (chunk + "x" * n)[:n]
        chunks.append(chunk)
        idx += n
    return ".".join(chunks)  # a rest between words


def compile_packet(packet: dict, melody, bpm=120, flow_pattern=None, delivery=None):
    """Compile a vox-lyric packet into a validated TONGUE score.

    (a) Syllables/phones come FROM the packet's ``words`` (G2P is never re-run). (b) When
    ``flow_pattern`` is None and the delivery is ``percussive``, placement is derived from each line's
    ``flow_hint`` (stress pattern onto the suggested grid, a rest between words); otherwise even
    quarter-note spread. (c) A line whose verdict is ``"rewrite"`` is REFUSED (``ValueError``
    naming the line) — the lyric gates finally bite. (d) A FLOW onset count that disagrees with
    the syllable count is a hard error (no silent cycle/drop).
    """
    melody = list(melody)
    if not melody:
        raise ValueError("melody must have at least one note")
    delivery = delivery or packet.get("delivery")
    lines = packet.get("lines", [])

    for ln in lines:  # (c) the gates bite: refuse rewrite-verdict lines
        if ln.get("verdict") == "rewrite":
            raise ValueError(
                f"compile_packet: line {ln.get('text')!r} has verdict 'rewrite' — refusing "
                f"(rewrite the line or drop it before synthesis)")

    atoms = []
    for ln in lines:
        atoms.extend(_packet_atoms(ln))

    use_flow = bool(flow_pattern) or (flow_pattern is None and delivery == "percussive")
    syllables = []

    if use_flow and atoms:
        from vox_flow.flow import compile_flow  # sibling tool, resolved on PYTHONPATH

        if flow_pattern:
            pattern = flow_pattern
            grid = 4
        else:  # derive from flow_hint per line, a rest between lines too
            grid = next((ln.get("flow_hint", {}).get("suggested_grid")
                         for ln in lines if ln.get("flow_hint", {}).get("suggested_grid")), None) or 4
            pattern = ".".join(_stress_line_pattern(ln) for ln in lines)

        flow_score = compile_flow(pattern, bpm=bpm, grid=int(grid),
                                  syllables=[a["text"] for a in atoms])
        events = flow_score["events"]
        if len(events) != len(atoms):  # (d) lyric-integrity hard error
            raise ValueError(
                f"compile_packet: FLOW pattern has {len(events)} onsets but {len(atoms)} "
                f"syllables (pattern {pattern!r}) — placement would drop/cycle syllables")
        spb = 60.0 / float(bpm)
        for i, ev in enumerate(events):
            note = melody[i % len(melody)]
            dyn = 1.0 if ev.get("accent") else 0.62
            syllables.append(_syl(atoms[i], note, start_beat=ev["t"] / spb,
                                  dur_beats=ev["dur"] / spb, dyn=dyn, articulation="spat"))
    else:
        for i, atom in enumerate(atoms):
            note = melody[i % len(melody)]
            syllables.append(_syl(atom, note, start_beat=float(i), dur_beats=1.0))

    meta = {"bpm": float(bpm), "key": None, "title": None}
    return load_score({"meta": meta, "syllables": syllables})
