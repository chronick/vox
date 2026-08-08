"""render — the deterministic TONGUE render path: say-render each syllable, then WORLD-impose
its exact per-syllable pitch, gate to its beat duration, and place it on the timeline.

Apple's TUNE format is dead on modern macOS, so we DON'T ask `say` to sing. Instead `say` is only
the glottal source (a real, reliably-voiced clip — honours the no-own-voice constraint), and the
sibling `vox_larynx` WORLD vocoder replaces the F0 skeleton with a constant target Hz while
keeping the formants (no chipmunking). That say->WORLD chain is the proven deterministic pitch
path. `vox_larynx.world` and its ``note_to_hz`` are resolved on PYTHONPATH.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .schema import load_score


def say_available() -> bool:
    return shutil.which("say") is not None


def note_to_hz_any(note):
    """Resolve a score ``note`` (note-name str, bare Hz, or numeric string) to Hz, or ``None``."""
    if note is None:
        return None
    if isinstance(note, (int, float)) and not isinstance(note, bool):
        return float(note)
    s = str(note).strip()
    try:
        return float(s)  # a bare-Hz string
    except ValueError:
        from vox_larynx.world import note_to_hz  # sibling tool, resolved on PYTHONPATH

        return note_to_hz(s)


def _say_render(text: str, voice: str, sr: int) -> np.ndarray:
    """Render one syllable to a mono float64 signal via macOS `say`."""
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="vox-tongue-") as tmp:
        out_path = str(Path(tmp) / "syl.wav")
        cmd = ["say", "-v", voice, "--file-format=WAVE", f"--data-format=LEI16@{sr}",
               "-o", out_path, text]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not Path(out_path).exists():
            raise RuntimeError(f"say failed for {text!r} (voice {voice!r}): "
                               f"{proc.stderr.decode('utf-8', 'replace')[:200]}")
        data, actual_sr = sf.read(out_path, dtype="float64", always_2d=True)
        if data.shape[0] == 0:
            raise RuntimeError(
                f"say produced no audio for {text!r} (voice {voice!r}, requested {sr} Hz)"
            )
        if int(actual_sr) != int(sr):
            raise RuntimeError(
                f"say rendered {text!r} at {actual_sr} Hz instead of requested {sr} Hz"
            )
    return data.mean(axis=1)


def _gate(clip: np.ndarray, dur_samples: int, fade_ms: float, sr: int) -> np.ndarray:
    """Hard-gate a clip to ``dur_samples`` with a short fade-out (no clicks)."""
    dur_samples = max(dur_samples, 1)
    clip = clip[:dur_samples] if len(clip) > dur_samples else clip
    fade = min(int(sr * fade_ms / 1000.0), len(clip))
    if fade > 1:
        clip = clip.copy()
        clip[-fade:] *= np.linspace(1.0, 0.0, fade)
    return clip


def render(score, sr: int = 44100, voice: str = "Fred"):
    """Render a TONGUE score to ``(samples float32 mono, manifest)``.

    Per syllable: `say`-render its text (cached per ``(text, voice)``); WORLD-impose the syllable's
    note Hz onto that clip (formants kept); gate to ``dur_beats*60/bpm`` seconds with an 8 ms fade;
    place at ``start_beat*60/bpm``; sum overlaps; peak-normalise to 0.9. ``manifest`` is a
    per-syllable list of ``{text, phones, t, dur, hz}``.
    """
    if not say_available():
        raise RuntimeError("`say` not found on PATH (macOS speech synth required to render TONGUE)")
    from vox_larynx import world  # sibling tool, resolved on PYTHONPATH

    score = load_score(score)
    bpm = score["meta"]["bpm"]
    spb = 60.0 / float(bpm)  # seconds per beat

    say_cache: dict[tuple, np.ndarray] = {}
    placed = []  # (start_sample, gated_signal)
    manifest = []
    for syl in score["syllables"]:
        text, voice_key = syl["text"], (syl["text"], voice)
        if voice_key not in say_cache:
            say_cache[voice_key] = _say_render(text, voice, sr)
        clip = say_cache[voice_key]

        hz = note_to_hz_any(syl["note"])
        if hz is not None:
            voiced, _ = world.render(clip, sr, to_hz=hz)
            voiced = np.asarray(voiced, dtype="float64")
        else:
            voiced = clip

        dur_s = syl["dur_beats"] * spb
        gated = _gate(voiced, int(round(dur_s * sr)), 8.0, sr)
        gated = gated * float(syl.get("dyn", 1.0))
        start = int(round(syl["start_beat"] * spb * sr))
        placed.append((start, gated))
        manifest.append({
            "text": text,
            "phones": syl["phones"],
            "t": float(syl["start_beat"] * spb),
            "dur": float(dur_s),
            "hz": float(hz) if hz is not None else None,
        })

    total = max((start + len(g) for start, g in placed), default=0) + 1
    out = np.zeros(total, dtype="float64")
    for start, g in placed:
        end = min(start + len(g), total)
        out[start:end] += g[: end - start]

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out = out / peak * 0.9
    return out.astype("float32"), manifest
