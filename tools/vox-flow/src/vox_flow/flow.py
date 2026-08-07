"""FLOW — a rhythmic cadence grammar: a syllable pattern DSL → grid-placed onset times.

Percussive vocal delivery lives *on* the grid — clipped syllables on 16ths, triplet
bursts, pushes and drags. FLOW makes the placement **authored** (a pattern) instead of
proportional (an even spread). The compiler is deterministic and audio-free, so it is
unit-testable and drives any downstream renderer (`say`, DiffSinger, a synth tract).

Pattern grammar (one char per grid step):
    x   onset (normal)          . or space   rest
    X   onset (accented)        - or _        hold/tie (extends the previous onset)

Modifiers: ``bpm``, ``grid`` (steps per beat: 4 = 16ths, 3 = 8th-triplets, 6 = 16th-triplets),
``swing`` (0–1, delays the off-steps), ``push_ms`` (global micro-timing; negative = rush/ahead,
positive = drag/behind). Syllables are assigned to onsets in order (cycled if fewer than onsets).
"""

from __future__ import annotations

ONSET = {"x", "X"}
ACCENT = {"X"}
REST = {".", " "}
HOLD = {"-", "_"}


def step_seconds(bpm: float, grid: int) -> float:
    """Duration of one grid step. ``grid`` = steps per quarter-note beat."""
    return 60.0 / float(bpm) / float(grid)


def compile_flow(pattern: str, *, bpm: float = 140.0, grid: int = 4, swing: float = 0.0,
                 push_ms: float = 0.0, syllables=None):
    """Compile a FLOW pattern into a list of onset events (the FLOW score).

    Each event: ``{step, t, dur, accent, syllable}`` — ``t``/``dur`` in float seconds. A HOLD
    extends the preceding onset's ``dur`` by one step (legato tie). Swing delays odd (off-beat)
    steps by ``swing × ½ step``. ``push_ms`` shifts every onset (rush/drag). Deterministic.
    """
    if not 0.0 <= swing <= 1.0:
        raise ValueError("swing must be in [0, 1]")
    step = step_seconds(bpm, grid)
    push = push_ms / 1000.0
    syllables = list(syllables) if syllables else []

    events = []
    onset_count = 0
    for i, ch in enumerate(pattern):
        if ch in HOLD and events:
            events[-1]["dur"] += step  # tie to previous onset
            continue
        if ch in ONSET:
            swing_off = (swing * 0.5 * step) if (i % 2 == 1) else 0.0
            t = i * step + swing_off + push
            syl = syllables[onset_count % len(syllables)] if syllables else None
            events.append({"step": i, "t": round(t, 5), "dur": round(step, 5),
                           "accent": ch in ACCENT, "syllable": syl})
            onset_count += 1
        # REST or unknown → nothing
    total = round(len(pattern) * step, 5)
    return {"events": events, "n_onsets": len(events), "bar_seconds": total,
            "bpm": bpm, "grid": grid, "swing": swing, "push_ms": push_ms}


# The grit chain: an ffmpeg filtergraph. Band-limit (telephone grit) → hard grid-gate
# (spits, no legato) → saturation-adjacent squash → comb/slap (metallic short echo). Applied to
# the assembled spat vocal. Kept as data so the chain is auditable + swappable.
GRIT_CHAIN = (
    "highpass=f=180,lowpass=f=3400,"                 # band-limit — the gritty midrange
    "agate=threshold=0.03:ratio=4:attack=1:release=40,"  # grid-gate — clip the tails, spit each syllable
    "acompressor=threshold=-16dB:ratio=6:attack=2:release=60,"  # squash dynamics onto the grid
    "aecho=0.7:0.4:11:0.35,"                          # comb/slap — short metallic reflection
    "alimiter=limit=0.95"                             # keep it hot but contained
)

# The bass chain: the grit treatment for a DEEP carrier voice. The grit chain's band-limit was
# authored for a mid-range spit — its highpass=180 guts a 55 Hz fundamental and its lowpass=3400
# deletes the sibilance band that carries injected consonants. This variant keeps the grid-gate /
# squash / comb-slap / limiter but relaxes the band-limit — highpass=55 (let the sub through) and
# lowpass=8000 (keep the ess band). Mid grit comes from a broad dip at 750 Hz (a scoop, not a hard
# cliff, and kept OFF the F2 decade) plus a presence bell at 2.6 kHz for consonant place cues.
BASS_CHAIN = (
    "highpass=f=55,lowpass=f=8000,"                  # keep the sub fundamental AND the ess band
    "equalizer=f=750:t=q:w=1:g=-2,"                  # grit scoop, off the F2 decade
    "equalizer=f=2600:t=q:w=1:g=3,"                  # presence bell — the consonant place-cue region
    "agate=threshold=0.03:ratio=4:attack=1:release=40,"  # grid-gate — clip the tails, spit each syllable
    "acompressor=threshold=-16dB:ratio=6:attack=2:release=60,"  # squash dynamics onto the grid
    "aecho=0.7:0.4:11:0.35,"                          # comb/slap — short metallic reflection
    "alimiter=limit=0.95"                             # keep it hot but contained
)

# Named chain registry: the CLI's --chain values resolve here; raw filtergraph strings also work.
CHAINS = {"grit": GRIT_CHAIN, "bass": BASS_CHAIN}


def accent_gain(accent: bool) -> float:
    """Velocity for an onset: accents punch, ghosts sit back."""
    return 1.0 if accent else 0.62
