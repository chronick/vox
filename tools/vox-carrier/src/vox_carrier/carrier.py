"""The carrier voice — a dry word-carrying signal poured into a deep harsh body.

The percussive spit alone (``vox flow render``) is thin and mid-forward. The carrier
keeps the FLOW *articulation* but gives it a body:

    modulator  = render_flow(compile_flow(...))     — the DRY say-spit (no chain): pure cadence
    body       = registry.render_body(<bass body>)  — a 55–62 Hz harsh voice (FM growl, sub-saw,
                                                      giant throat): the timbre, the "deep"
    vocode     = apply_vocode(body, modulator)      — the body SPEAKS the modulator's syllables
    BASS_CHAIN                                       — grid-gate + slap, band-relaxed to keep the sub

Two defaults here are load-bearing, not options:

* ``apply_vocode(lo_hz=40.0, hi_hz=8000.0)`` — a higher lo_hz would band-pass the body's own
  55 Hz fundamental straight out (the "deep" removed by the very op meant to deliver it);
  40 Hz keeps it, and 40–8k packs more bands into the 300–3k articulation decade.
* the ``bass`` chain (vox_flow.flow.BASS_CHAIN) — highpass=55 / lowpass=8000 instead of the
  grit chain's 180/3400, so neither the sub fundamental nor the consonant-carrying ess band
  is deleted.

The whole path is deterministic given the machine's `say` voice and (for SC bodies) sclang;
all audio I/O routes through the CAS so the vocoder sees real frames and lineage is auditable.
"""

from __future__ import annotations

import io

import numpy as np
from vox_flow.flow import compile_flow
from vox_flow.render import apply_chain, render_flow


def _cas_samples(samples: np.ndarray, sr: int, *, role: str) -> dict:
    """CAS a mono float array as a WAV blob and mint the matching `audio` frame.

    The vocoder (``smpl_analysis.duo.apply_vocode``) resolves its inputs from CAS by ``hash``,
    so the raw render arrays have to be materialized as real frames before they can be vocoded.
    """
    import soundfile as sf
    from smplstream import cas
    from smplstream import frames as F

    arr = np.ascontiguousarray(np.asarray(samples, dtype="float32"))
    if arr.ndim == 1:
        arr = arr[:, None]
    buf = io.BytesIO()
    sf.write(buf, arr, int(sr), format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    return F.audio_frame(
        h,
        sr=meta.get("sr", int(sr)),
        ch=meta.get("ch", arr.shape[1]),
        dur=meta.get("dur", arr.shape[0] / sr if sr else 0.0),
        role=role,
        fmt=meta.get("fmt"),
    )


def _flatten_body(y: np.ndarray, sr: int, win_ms: float = 60.0) -> np.ndarray:
    """Whiten a body's *amplitude* — divide out its short-time RMS so only the modulator's
    envelope shapes the vocoder output.

    A channel vocoder imprints the MODULATOR's dynamics onto the BODY's timbre; but if the body
    carries its own loudness rhythm (a re-voiced speech clip has the source's syllables and
    pauses baked in), that rhythm competes with the modulator's grid and smears the spat onsets.
    Dividing by a smooth RMS envelope turns the body into a steady drone at constant level — the
    sustained bass the FLOW cadence then articulates. A no-op in effect for the already-flat
    synth bodies (growl / sub-saw).
    """
    x = np.ascontiguousarray(y, dtype="float64")
    w = max(int(win_ms / 1000.0 * sr), 1)
    rms = np.sqrt(np.convolve(x * x, np.ones(w) / w, mode="same") + 1e-9)
    floor = 1e-3 * float(np.max(rms) or 1.0)          # don't divide up pure-silence regions to noise
    flat = x / (rms + floor)
    peak = float(np.max(np.abs(flat))) or 1.0
    return (flat / peak * 0.9).astype("float64")


def _resolve_mono(frame: dict) -> tuple[np.ndarray, int]:
    """Resolve an audio frame's CAS blob back to ``(mono float64, sr)``."""
    import soundfile as sf
    from smplstream import cas

    data, sr = sf.read(str(cas.get_path(frame["hash"])), dtype="float64", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(mono, dtype="float64"), int(sr)


def render_carrier_one(
    pattern: str,
    syllables,
    bpm: float,
    body_name: str,
    *,
    sr: int = 44100,
    grid: int = 4,
    swing: float = 0.0,
    push_ms: float = 0.0,
    bands: int = 24,
    ess_mix: float = 0.30,
    flatten_body: bool = True,
    bodies_path=None,
) -> dict:
    """Render one carrier take: a FLOW cadence spoken by a deep harsh body.

    Pipeline: FLOW dry modulator (``render_flow``, no chain) → body (``registry.render_body``,
    rendered ``modulator_dur + 0.3`` s so it never runs out under the spit) → optional body
    amplitude-whitening → channel vocode (``lo_hz=40 / hi_hz=8000`` band grid) → the bass chain.

    On a deep body a large ``ess_mix`` makes the sibilant bursts dominate the mix and the global
    peak-normalize then crushes the vowel onsets, weakening the of-the-grid feel — tune it low
    when the grid matters more than consonant sparkle. ``flatten_body`` whitens the body's own
    amplitude so only the modulator shapes the dynamics. Returns a dict with the final take plus
    every intermediate frame/array for measurement::

        {"final": float32 mono, "sr": int, "modulator": float32 mono (dry),
         "wet": float32 mono (vocoded, pre-chain), "body_frame", "modulator_frame",
         "wet_frame", "params": {...}}
    """
    from smpl_analysis.duo import apply_vocode
    from vox_bodies import registry

    score = compile_flow(pattern, bpm=bpm, grid=grid, swing=swing, push_ms=push_ms,
                         syllables=list(syllables))
    modulator = render_flow(score, sr=sr)                       # DRY — no chain
    mod_dur = len(modulator) / float(sr)

    body, body_sr = registry.render_body(body_name, dur=mod_dur + 0.3, sr=sr, path=bodies_path)
    if flatten_body:
        body = _flatten_body(body, body_sr)

    mod_frame = _cas_samples(modulator, sr, role="carrier.modulator")
    body_frame = _cas_samples(body, body_sr, role="carrier.body")

    wet_frame = apply_vocode(body_frame, mod_frame, bands=bands,
                             lo_hz=40.0, hi_hz=8000.0, ess_mix=ess_mix)
    wet, wet_sr = _resolve_mono(wet_frame)

    final = apply_chain(wet.astype("float32"), wet_sr, "bass")

    return {
        "final": final,
        "sr": wet_sr,
        "modulator": modulator,
        "modulator_sr": sr,
        "wet": wet.astype("float32"),
        "body_frame": body_frame,
        "modulator_frame": mod_frame,
        "wet_frame": wet_frame,
        "params": {
            "pattern": pattern,
            "syllables": list(syllables),
            "bpm": float(bpm),
            "grid": grid,
            "body": body_name,
            "bands": bands,
            "vocode_lo_hz": 40.0,
            "vocode_hi_hz": 8000.0,
            "ess_mix": float(ess_mix),
            "flatten_body": bool(flatten_body),
            "chain": "bass",
            "n_onsets": score["n_onsets"],
            "bar_seconds": score["bar_seconds"],
        },
    }


# ---------------------------------------------------------------------------
# Verse composite — a full spat verse (one bar per lyric line) poured into a
# deep body, plus the dry-diction layer.
#
# render_carrier_one renders ONE authored FLOW pattern. render_carrier_verse
# takes real lyric LINES, derives each line's grid pattern from its packet
# flow_hint stress pattern (X/x onto the grid, a rest between words), renders
# each line's dry spat bar, concatenates them into one modulator, and vocodes a
# single body across the whole verse. The composite adds the DRY-DICTION LAYER:
# a quiet band-limited copy of the dry words mixed under the vocoded body so
# the diction survives the formant-starved bass.
# ---------------------------------------------------------------------------


def _bandpass(x: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> np.ndarray:
    """4th-order Butterworth band-pass (SOS). Used for the dry-diction layer (~0.5–5 kHz — the
    band that carries diction without adding sub-mud or hiss)."""
    from scipy.signal import butter, sosfilt

    nyq = sr / 2.0
    lo = max(float(lo_hz) / nyq, 1e-5)
    hi = min(float(hi_hz) / nyq, 0.999)
    if hi <= lo:
        return np.zeros_like(x, dtype="float64")
    sos = butter(4, [lo, hi], btype="bandpass", output="sos")
    return sosfilt(sos, np.ascontiguousarray(x, dtype="float64"))


def _rms(x: np.ndarray) -> float:
    x = np.ascontiguousarray(x, dtype="float64")
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def line_flow_score(line_text: str, bpm: float, *, grid: int = 4) -> dict | None:
    """Compile ONE lyric line into a FLOW score: pattern from the lyric packet's flow_hint stress
    pattern (a rest between words, via the score compiler's ``_stress_line_pattern``), syllable
    texts from ``vox_tongue.g2p.syllabify_line``. Returns ``None`` for a blank/dropped line.

    Onset-count and text-count are guaranteed equal for in-CMUdict lines (both use the shared
    one-vowel-nucleus syllable contract); a guard pads/truncates the texts on the rare heuristic
    disagreement so a syllable is never silently cycled or dropped onto the wrong onset.
    """
    from vox_lyric.packet import build_packet
    from vox_tongue.compile import _stress_line_pattern
    from vox_tongue.g2p import syllabify_line

    pk = build_packet([line_text], "percussive")
    if not pk["lines"]:
        return None
    ln = pk["lines"][0]
    pattern = _stress_line_pattern(ln)
    n_onsets = sum(1 for c in pattern if c in ("x", "X"))
    if n_onsets == 0:
        return None

    texts = [t for w in syllabify_line(line_text) for t in w["syllable_texts"]]
    if len(texts) != n_onsets:  # heuristic-disagreement guard (never drop/cycle a syllable)
        if not texts:
            texts = ["uh"]
        if len(texts) < n_onsets:
            texts = texts + [texts[-1]] * (n_onsets - len(texts))
        else:
            texts = texts[:n_onsets]

    return compile_flow(pattern, bpm=bpm, grid=grid, syllables=texts)


def render_verse_modulator(lines, bpm: float, *, sr: int = 44100, grid: int = 4) -> tuple:
    """Render every lyric line to its DRY spat bar (``render_flow``, no chain) and concatenate
    into one modulator. Returns ``(modulator float32, per_line_scores)`` — one bar per non-blank
    line, back-to-back (each bar already carries render_flow's short tail, so bars don't smear)."""
    bars = []
    scores = []
    for text in lines:
        score = line_flow_score(text, bpm, grid=grid)
        if score is None:
            continue
        bars.append(render_flow(score, sr=sr))
        scores.append(score)
    if not bars:
        raise ValueError("render_verse_modulator: no renderable lines")
    modulator = np.concatenate(bars).astype("float32")
    return modulator, scores


def _tile_to_length(y: np.ndarray, n: int) -> np.ndarray:
    """Loop-tile a body array to at least ``n`` samples, then trim. Needed for fixed-length
    clip-derived bodies under a long verse; a no-op when an SC body is already long enough
    (SC bodies render to any requested ``dur`` cheaply)."""
    y = np.ascontiguousarray(y, dtype="float64")
    if y.size == 0:
        return np.zeros(n, dtype="float64")
    if y.size >= n:
        return y[:n]
    reps = int(np.ceil(n / y.size))
    return np.tile(y, reps)[:n]


def render_carrier_verse(
    lines,
    bpm: float,
    body_name: str,
    *,
    sr: int = 44100,
    grid: int = 4,
    bands: int = 24,
    ess_mix: float = 0.30,
    dry_db: float | None = -14.0,
    dry_band=(500.0, 5000.0),
    flatten_body: bool = True,
    bodies_path=None,
) -> dict:
    """Render a full carrier verse: real lyric ``lines`` spat on the grid, poured into
    ``body_name``, with the dry-diction layer mixed under.

    Pipeline: per-line dry FLOW bars concatenated → one modulator → body rendered/loop-tiled to
    the modulator length → optional amplitude-whitening → channel vocode (``lo_hz=40 /
    hi_hz=8000`` grid, ``ess_mix``) → the bass chain = the vocoded BODY → **dry-diction layer**:
    the dry modulator band-passed to ``dry_band`` (default 500–5000 Hz) and mixed under the body
    at ``dry_db`` dB relative to the body's RMS (default −14 dB; ``None`` disables the layer).
    Peak-safe-limited to 0.95.

    Returns ``{final, sr, modulator, modulator_sr, wet, body, params}`` — ``body`` is the
    vocoded take BEFORE the dry layer (so a caller/test can measure the layer's lift from one
    render), ``final`` is body + dry layer.
    """
    modulator, scores = render_verse_modulator(lines, bpm, sr=sr, grid=grid)
    final, body, wet, wet_sr = _vocode_and_layer(
        modulator, sr, body_name, bands=bands, ess_mix=ess_mix, dry_db=dry_db,
        dry_band=dry_band, flatten_body=flatten_body, bodies_path=bodies_path,
        role="carrier.verse",
    )
    return {
        "final": final,
        "sr": wet_sr,
        "modulator": modulator,
        "modulator_sr": sr,
        "wet": wet,
        "body": body,
        "params": {
            "n_lines": len(scores),
            "bpm": float(bpm),
            "grid": grid,
            "body": body_name,
            "bands": bands,
            "vocode_lo_hz": 40.0,
            "vocode_hi_hz": 8000.0,
            "ess_mix": float(ess_mix),
            "dry_db": (None if dry_db is None else float(dry_db)),
            "dry_band_hz": [float(dry_band[0]), float(dry_band[1])],
            "flatten_body": bool(flatten_body),
            "chain": "bass",
            "mod_seconds": round(len(modulator) / float(sr), 4),
        },
    }


def _vocode_and_layer(
    modulator: np.ndarray,
    sr: int,
    body_name: str,
    *,
    bands: int = 24,
    ess_mix: float = 0.30,
    dry_db: float | None = -14.0,
    dry_band=(500.0, 5000.0),
    flatten_body: bool = True,
    bodies_path=None,
    role: str = "carrier",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Voice a DRY modulator through the deep body: body render/loop-tile → amplitude-whiten →
    channel vocode (``lo_hz=40 / hi_hz=8000`` grid) → the bass chain = the vocoded BODY →
    dry-diction layer band-passed to ``dry_band`` and mixed under at ``dry_db`` dB rel. body RMS
    (``None`` disables it). Returns ``(final f32, body f32, wet f32, wet_sr)`` — ``body`` is the
    take BEFORE the dry layer, ``final`` = body + layer, peak-safe-limited to 0.95."""
    from smpl_analysis.duo import apply_vocode
    from vox_bodies import registry

    modulator = np.ascontiguousarray(modulator, dtype="float32")
    mod_dur = len(modulator) / float(sr)

    body, body_sr = registry.render_body(body_name, dur=mod_dur + 0.3, sr=sr, path=bodies_path)
    need = int((mod_dur + 0.3) * body_sr)
    body = _tile_to_length(body, need)                # loop-tile short clip-derived bodies
    if flatten_body:
        body = _flatten_body(body, body_sr)

    mod_frame = _cas_samples(modulator, sr, role=f"{role}.modulator")
    body_frame = _cas_samples(body, body_sr, role=f"{role}.body")

    wet_frame = apply_vocode(body_frame, mod_frame, bands=bands,
                             lo_hz=40.0, hi_hz=8000.0, ess_mix=ess_mix)
    wet, wet_sr = _resolve_mono(wet_frame)

    voiced = apply_chain(wet.astype("float32"), wet_sr, "bass")

    # The dry-diction layer: mix a quiet, band-limited copy of the DRY words under the vocoded
    # body so the diction survives the formant-starved bass.
    body64 = np.ascontiguousarray(voiced, dtype="float64")
    if dry_db is None:
        final = body64.copy()
    else:
        # resample the dry modulator to the body's sr if they differ (they don't by default).
        mod64 = np.ascontiguousarray(modulator, dtype="float64")
        if sr != wet_sr:
            from math import gcd

            from scipy.signal import resample_poly
            g = gcd(int(wet_sr), int(sr)) or 1
            mod64 = resample_poly(mod64, wet_sr // g, sr // g)
        dry = _bandpass(mod64, wet_sr, float(dry_band[0]), float(dry_band[1]))
        body_rms, dry_rms = _rms(body64), _rms(dry)
        n = min(len(body64), len(dry))
        final = body64[:n].copy()
        if dry_rms > 1e-9 and body_rms > 1e-9:
            gain = (body_rms * (10.0 ** (dry_db / 20.0))) / dry_rms
            final += gain * dry[:n]

    peak = float(np.max(np.abs(final))) if final.size else 0.0
    if peak > 0.95:
        final = final * (0.95 / peak)
    return (final.astype("float32"), np.asarray(voiced, dtype="float32"),
            wet.astype("float32"), wet_sr)


def place_clips_on_grid(
    clips,
    sr: int,
    bpm: float,
    *,
    bars_per_line: int = 2,
    beats_per_bar: int = 4,
    fade_ms: float = 4.0,
) -> tuple[np.ndarray, list[int]]:
    """Assemble per-line clips into one verse modulator by placing each clip at its line's grid
    boundary — ``bars_per_line`` bars apart — while preserving the natural flow *inside* each clip
    (no syllable chopping; the light grid feel comes from where lines START, not from cutting them).

    Every clip gets a short linear edge fade (``fade_ms``, clamped to half the clip) so both ends
    reach zero; placement is then a plain overlap-add. That makes seams click-free even when a clip
    over-runs its slot into the next line (the tail fades out as the next head fades in) and when
    a clip is shorter than its slot (the gap is true silence, a natural breath). Returns
    ``(modulator float32 mono, starts_samples)``.
    """
    clips = [np.ascontiguousarray(np.asarray(c, dtype="float32")).reshape(-1) for c in clips]
    if not clips:
        return np.zeros(0, dtype="float32"), []

    slot = round(bars_per_line * beats_per_bar * 60.0 / float(bpm) * sr)
    if slot <= 0:
        raise ValueError(f"non-positive slot ({slot}) from bpm={bpm}, bars_per_line={bars_per_line}")
    starts = [i * slot for i in range(len(clips))]

    def _edge_faded(c: np.ndarray) -> np.ndarray:
        if c.size == 0:
            return c
        f = min(int(fade_ms / 1000.0 * sr), c.size // 2)
        if f <= 0:
            return c
        c = c.copy()
        ramp = np.linspace(0.0, 1.0, f, endpoint=False, dtype="float32")
        c[:f] *= ramp
        c[-f:] *= ramp[::-1]
        return c

    n_out = max(s + c.size for s, c in zip(starts, clips))
    out = np.zeros(n_out, dtype="float32")
    for s, c in zip(starts, clips):
        cf = _edge_faded(c)
        out[s:s + cf.size] += cf
    return out, starts


def render_carrier_modulated(
    modulator: np.ndarray,
    body_name: str,
    *,
    sr: int = 44100,
    bands: int = 24,
    ess_mix: float = 0.30,
    dry_db: float | None = -14.0,
    dry_band=(500.0, 5000.0),
    flatten_body: bool = True,
    bodies_path=None,
) -> dict:
    """Render a carrier take from a PRE-BUILT dry ``modulator`` (e.g. neural-TTS line clips
    assembled by :func:`place_clips_on_grid`) instead of a FLOW say-spit — keeping the deep body
    + bass chain and only swapping the skeleton. Same voicing core and return contract as
    :func:`render_carrier_verse`: ``{final, sr, modulator, modulator_sr, wet, body, params}``."""
    modulator = np.ascontiguousarray(modulator, dtype="float32")
    final, body, wet, wet_sr = _vocode_and_layer(
        modulator, sr, body_name, bands=bands, ess_mix=ess_mix, dry_db=dry_db,
        dry_band=dry_band, flatten_body=flatten_body, bodies_path=bodies_path,
        role="carrier.external",
    )
    return {
        "final": final,
        "sr": wet_sr,
        "modulator": modulator,
        "modulator_sr": sr,
        "wet": wet,
        "body": body,
        "params": {
            "skeleton": "external",
            "body": body_name,
            "bands": bands,
            "vocode_lo_hz": 40.0,
            "vocode_hi_hz": 8000.0,
            "ess_mix": float(ess_mix),
            "dry_db": (None if dry_db is None else float(dry_db)),
            "dry_band_hz": [float(dry_band[0]), float(dry_band[1])],
            "flatten_body": bool(flatten_body),
            "chain": "bass",
            "mod_seconds": round(len(modulator) / float(sr), 4),
        },
    }
