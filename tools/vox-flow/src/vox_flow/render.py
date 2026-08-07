"""FLOW render — place `say`-rendered syllables on the FLOW grid, then an optional fx chain.

`say` is the glottal source (a synthetic system voice — no recorded human source), placed
sample-accurately at each onset, hard-gated to its step duration (the spat, of-the-grid feel),
accent-weighted, mixed, and optionally driven through a named ffmpeg chain. Deterministic given
the machine's `say` voice; `say`/`ffmpeg` are system binaries, gated by PATH discovery.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .flow import CHAINS, accent_gain


def say_available() -> bool:
    return shutil.which("say") is not None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _say_syllable(text: str, voice: str, pbas: float | None, sr: int, out_path: str) -> None:
    """Render one syllable to a WAV via macOS `say`. `pbas` sets the source pitch base (Hz→scale)."""
    payload = f"[[pbas {pbas}]]{text}" if pbas else text
    cmd = ["say", "-v", voice, "--file-format=WAVE", f"--data-format=LEI16@{sr}", "-o", out_path, payload]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not Path(out_path).exists():
        raise RuntimeError(f"say failed for {text!r} (voice {voice!r}): "
                           f"{proc.stderr.decode('utf-8', 'replace')[:200]}")


def _gate(clip: np.ndarray, dur_samples: int, fade_ms: float, sr: int) -> np.ndarray:
    """Hard-gate a syllable to ``dur_samples`` (the spat feel) with a short fade-out (no clicks)."""
    clip = clip[:dur_samples] if len(clip) > dur_samples else clip
    fade = min(int(sr * fade_ms / 1000.0), len(clip))
    if fade > 1:
        clip = clip.copy()
        clip[-fade:] *= np.linspace(1.0, 0.0, fade)
    return clip


def render_flow(score: dict, *, voice: str = "Fred", pbas: float | None = 92.0, sr: int = 44100,
                tail_s: float = 0.25, gate_fade_ms: float = 8.0) -> np.ndarray:
    """Render a compiled FLOW score to a mono float32 spat-vocal signal (pre-chain)."""
    if not say_available():
        raise RuntimeError("`say` not found on PATH (macOS speech synth required for FLOW render)")
    events = [e for e in score["events"] if e.get("syllable")]
    n = int((score["bar_seconds"] + tail_s) * sr) + 1
    out = np.zeros(n, dtype="float64")

    with tempfile.TemporaryDirectory(prefix="vox-flow-") as tmp:
        cache: dict[str, np.ndarray] = {}
        for ev in events:
            syl = ev["syllable"]
            if syl not in cache:
                import soundfile as sf
                p = str(Path(tmp) / f"syl_{abs(hash(syl))}.wav")
                _say_syllable(syl, voice, pbas, sr, p)
                data, _ = sf.read(p, dtype="float64", always_2d=True)
                cache[syl] = data.mean(axis=1)
            clip = _gate(cache[syl], int(ev["dur"] * sr), gate_fade_ms, sr)
            clip = clip * accent_gain(ev["accent"])
            start = int(ev["t"] * sr)
            end = min(start + len(clip), n)
            out[start:end] += clip[: end - start]

    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.9).astype("float32")


def apply_chain(x: np.ndarray, sr: int, chain: str = "grit") -> np.ndarray:
    """Drive a signal through an ffmpeg filtergraph. ``chain`` is a registry name from
    :data:`vox_flow.flow.CHAINS` or a raw filtergraph string. Returns processed mono float32."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH (required for the fx chain)")
    graph = CHAINS.get(chain, chain)
    import soundfile as sf
    with tempfile.TemporaryDirectory(prefix="vox-flow-fx-") as tmp:
        src, dst = str(Path(tmp) / "in.wav"), str(Path(tmp) / "out.wav")
        sf.write(src, x.astype("float32"), sr, subtype="FLOAT")
        cmd = ["ffmpeg", "-y", "-i", src, "-af", graph, "-c:a", "pcm_f32le", dst]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or not Path(dst).exists():
            raise RuntimeError(f"ffmpeg chain failed: {proc.stderr.decode('utf-8','replace')[-300:]}")
        data, _ = sf.read(dst, dtype="float64", always_2d=True)
    y = data.mean(axis=1)
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * 0.9).astype("float32")
