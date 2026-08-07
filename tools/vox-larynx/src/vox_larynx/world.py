"""WORLD-vocoder DSP core for the larynx tier — pure numpy/pyworld, no smplstream import.

Every function takes a mono float64 signal + sample rate and returns ``(samples, params)``
(or an analysis dict). The CLI layer handles CAS I/O and frame minting; this module is the
DSP and is unit-testable standalone. WORLD decomposes a voice into three streams:

    F0   — the fundamental-frequency contour (the *melody skeleton*; 0 = unvoiced)
    SP   — the spectral envelope (the *formants / body* — the vowel identity, the timbre)
    AP   — aperiodicity (the *breath / noise* balance per band)

The useful reframe: F0 is the melody skeleton, SP is the body. **Replace F0 and keep SP**
and you retune a voice to any pitch WITHOUT chipmunking (formants preserved), which is the
whole trick behind the harmonizer here and the pitch-imposition render path in the
phoneme-score tool.
"""

from __future__ import annotations

import numpy as np

# Bumped on ANY behaviour change (smplstream spec → Memoization). Folded into op_version
# alongside the pyworld version by the CLI.
ANALYZE_OP_VERSION = "larynx-analyze@1"
RENDER_OP_VERSION = "larynx-render@1"
HARMONIZE_OP_VERSION = "larynx-harmonize@1"

FRAME_PERIOD = 5.0  # ms — WORLD's default analysis/synthesis hop.
F0_FLOOR = 65.0     # Hz — below low male speech; voice material sits above this.
F0_CEIL = 1000.0    # Hz — above a soprano; leaves headroom for octave-up harmonies.

_A4_HZ = 440.0
_NOTE_SEMITONES = {"c": -9, "c#": -8, "db": -8, "d": -7, "d#": -6, "eb": -6, "e": -5,
                   "f": -4, "f#": -3, "gb": -3, "g": -2, "g#": -1, "ab": -1, "a": 0,
                   "a#": 1, "bb": 1, "b": 2}


def note_to_hz(note: str) -> float:
    """``"A3"`` / ``"C#4"`` / ``"Eb2"`` → Hz (A4 = 440). Case-insensitive."""
    s = note.strip().lower()
    i = 1
    if len(s) > 1 and s[1] in "#b":
        i = 2
    name, octave = s[:i], s[i:]
    if name not in _NOTE_SEMITONES:
        raise ValueError(f"bad note name: {note!r}")
    semis = _NOTE_SEMITONES[name] + (int(octave) - 4) * 12
    return _A4_HZ * 2.0 ** (semis / 12.0)


# ---------------------------------------------------------------------------
# WORLD analysis / synthesis.
# ---------------------------------------------------------------------------
def _prep(x: np.ndarray) -> np.ndarray:
    """WORLD needs a C-contiguous float64 mono vector."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype="float64")


def analyze(x: np.ndarray, sr: int):
    """Decompose into ``(f0, sp, ap, t)`` via harvest → cheaptrick → d4c."""
    import pyworld as pw

    x = _prep(x)
    f0, t = pw.harvest(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_PERIOD)
    f0 = pw.stonemask(x, f0, t, sr)  # refine harvest's F0 (reduces octave errors)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    return f0, sp, ap, t


def synth(f0: np.ndarray, sp: np.ndarray, ap: np.ndarray, sr: int) -> np.ndarray:
    import pyworld as pw

    y = pw.synthesize(np.ascontiguousarray(f0, dtype="float64"),
                      np.ascontiguousarray(sp, dtype="float64"),
                      np.ascontiguousarray(ap, dtype="float64"), sr, frame_period=FRAME_PERIOD)
    return y.astype("float64")


def f0_stats(f0: np.ndarray) -> dict:
    """Summary of the F0 contour — the melody skeleton's shape, for a `feature` frame."""
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return {"voiced_frac": 0.0, "f0_median_hz": None, "f0_min_hz": None,
                "f0_max_hz": None, "f0_range_semitones": None, "n_frames": int(f0.size)}
    med = float(np.median(voiced))
    lo, hi = float(np.min(voiced)), float(np.max(voiced))
    # Cast every value to a native Python scalar — a stray numpy float breaks JSON frame I/O.
    return {
        "voiced_frac": round(float((f0 > 0).mean()), 4),
        "f0_median_hz": round(med, 2),
        "f0_min_hz": round(lo, 2),
        "f0_max_hz": round(hi, 2),
        "f0_range_semitones": round(float(12.0 * np.log2(hi / lo)), 2) if lo > 0 else None,
        "n_frames": int(f0.size),
    }


# ---------------------------------------------------------------------------
# Spectral-envelope (formant) warp — the timbre / "otherworldliness" dial.
# ---------------------------------------------------------------------------
def _warp_formants(sp: np.ndarray, ratio: float) -> np.ndarray:
    """Shift the spectral envelope along frequency by ``ratio`` (>1 = formants up).

    Resamples each frame's envelope on the frequency axis: ``new[k] = sp[k / ratio]``. A ratio
    away from 1 detaches the vowel from any human vocal-tract length — the abstract-voice-box
    lever. ratio == 1 is an exact passthrough.
    """
    if abs(ratio - 1.0) < 1e-6:
        return sp
    n_bins = sp.shape[1]
    src_idx = np.arange(n_bins) / float(ratio)
    lo = np.clip(np.floor(src_idx).astype(int), 0, n_bins - 1)
    hi = np.clip(lo + 1, 0, n_bins - 1)
    frac = np.clip(src_idx - lo, 0.0, 1.0)
    return sp[:, lo] * (1.0 - frac) + sp[:, hi] * frac


def _time_scale(f0, sp, ap, ratio: float):
    """Resample the WORLD frame axis by ``ratio`` (>1 = longer/slower) — time-stretch that
    leaves pitch and formants untouched (independent of the F0 edit)."""
    if abs(ratio - 1.0) < 1e-6:
        return f0, sp, ap
    n = f0.shape[0]
    n2 = max(int(round(n * ratio)), 1)
    src = np.linspace(0, n - 1, n2)
    lo = np.floor(src).astype(int)
    hi = np.clip(lo + 1, 0, n - 1)
    fr = (src - lo)[:, None]
    f0v = f0.copy()
    f0v[f0v <= 0] = np.nan                       # interpolate only within voiced runs
    f0i = np.interp(src, np.arange(n), np.nan_to_num(f0v, nan=0.0))
    unvoiced = np.interp(src, np.arange(n), (f0 <= 0).astype(float)) > 0.5
    f0i[unvoiced] = 0.0
    spi = sp[lo] * (1 - fr) + sp[hi] * fr
    api = ap[lo] * (1 - fr) + ap[hi] * fr
    return f0i, spi, api


# ---------------------------------------------------------------------------
# render — replace / shift F0, warp formants, time-scale, resynth.
# ---------------------------------------------------------------------------
def render(x, sr, *, semitones=0.0, to_hz=None, flatten=False, formant_ratio=1.0,
           time_ratio=1.0):
    """Re-voice a signal: edit the F0 skeleton (keep SP/AP), optionally warp formants / time.

    Exactly one pitch mode applies, in precedence order:
      - ``to_hz`` set     → every voiced frame becomes that constant Hz (monotone target).
      - ``flatten``       → every voiced frame becomes the median F0 (dead-flat, same pitch).
      - else              → multiply F0 by ``2**(semitones/12)`` (a transposition).
    ``formant_ratio`` warps the vowel body; ``time_ratio`` stretches duration. This is the
    per-phoneme F0/duration imposition primitive the TONGUE compile path drives.
    """
    f0, sp, ap, _ = analyze(x, sr)
    voiced = f0 > 0
    if to_hz is not None:
        f0_new = np.where(voiced, float(to_hz), 0.0)
        mode = f"to_hz={to_hz}"
    elif flatten:
        med = float(np.median(f0[voiced])) if voiced.any() else 0.0
        f0_new = np.where(voiced, med, 0.0)
        mode = f"flatten@{round(med, 2)}"
    else:
        f0_new = f0 * (2.0 ** (float(semitones) / 12.0))
        mode = f"shift={semitones}st"

    sp = _warp_formants(sp, float(formant_ratio))
    f0_new, sp, ap = _time_scale(f0_new, sp, ap, float(time_ratio))
    y = synth(f0_new, sp, ap, sr)
    params = {"pitch_mode": mode, "formant_ratio": float(formant_ratio),
              "time_ratio": float(time_ratio), "sr_hz": sr,
              **{f"in_{k}": v for k, v in f0_stats(f0).items() if k in ("f0_median_hz", "voiced_frac")}}
    return y, params


# ---------------------------------------------------------------------------
# harmonize — chord-locked copies of one voice, formants preserved.
# ---------------------------------------------------------------------------
def _voice_f0(f0_lead, interval_semis, *, mode, root_hz, jitter, vib_cents, vib_rate,
              sr, phase, rng):
    """One harmony voice's F0 contour: parallel-follow the lead OR hold a flat chord tone,
    plus per-voice micro-jitter + a decorrelated LF vibrato (what turns N clones into a choir)."""
    voiced = f0_lead > 0
    if mode == "flat":                     # fixed-pitch sheen — each voice holds one pitch
        base = root_hz * 2.0 ** (interval_semis / 12.0)
        f0v = np.where(voiced, base, 0.0)
    else:                                  # follow — natural choir, parallel to the melody
        f0v = f0_lead * 2.0 ** (interval_semis / 12.0)

    n = f0v.shape[0]
    hop_s = FRAME_PERIOD / 1000.0
    t = np.arange(n) * hop_s
    if jitter:
        f0v = f0v * (1.0 + jitter * rng.standard_normal(n))
    if vib_cents:
        vib = 2.0 ** ((vib_cents / 1200.0) * np.sin(2.0 * np.pi * vib_rate * t + phase))
        f0v = f0v * vib
    f0v[~voiced] = 0.0
    return f0v


def harmonize(x, sr, *, chord=(0, 3, 7), mode="follow", include_lead=True, drone=False,
              drone_octave=-1, jitter=0.004, vib_cents=18.0, vib_rate=5.2, seed=0):
    """Stack ``chord`` (semitone offsets) as formant-preserving harmony voices.

    Each voice shares the lead's SP/AP (so no chipmunking — the vowel body is intact) but rides
    a chord-tone F0 contour, jittered and LF-vibrato'd on a decorrelated phase so N clones read
    as a choir rather than a flanged unison. ``mode='flat'`` holds each voice on one pitch
    (a robotic fixed-pitch sheen); ``'follow'`` tracks the melody in parallel (natural choir).
    ``drone`` adds a sustained tonic under everything. Returns the mixed stack.
    """
    rng = np.random.default_rng(seed)
    f0, sp, ap, _ = analyze(x, sr)
    voiced = f0 > 0
    root_hz = float(np.median(f0[voiced])) if voiced.any() else 220.0

    voices = []
    if include_lead:
        voices.append(synth(f0, sp, ap, sr))
    for k, interval in enumerate(chord):
        phase = 2.0 * np.pi * (k + 1) / (len(chord) + 1)   # decorrelate vibrato per voice
        f0v = _voice_f0(f0, interval, mode=mode, root_hz=root_hz, jitter=jitter,
                        vib_cents=vib_cents, vib_rate=vib_rate * (1.0 + 0.03 * k),
                        sr=sr, phase=phase, rng=rng)
        voices.append(synth(f0v, sp, ap, sr))
    if drone:
        drone_hz = root_hz * 2.0 ** (drone_octave)         # octave(s) below the tonic
        f0d = np.full_like(f0, drone_hz)                   # sustained through unvoiced gaps too
        voices.append(0.7 * synth(f0d, sp, ap, sr))

    length = max(v.shape[0] for v in voices)
    mix = np.zeros(length, dtype="float64")
    for v in voices:
        mix[:v.shape[0]] += v
    peak = float(np.max(np.abs(mix))) or 1.0
    mix = mix / peak * 0.9                                  # headroom; downstream limit finalises
    params = {"chord": list(chord), "mode": mode, "voices": len(voices),
              "include_lead": include_lead, "drone": drone, "root_hz": round(root_hz, 2),
              "jitter": jitter, "vib_cents": vib_cents, "vib_rate_hz": vib_rate,
              "seed": seed, "sr_hz": sr}
    return mix, params
