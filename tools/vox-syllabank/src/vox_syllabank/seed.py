"""Seed the bank from LICENSE-CLEAN self-renders — the only thing v0 admits.

Two sources, both ``license: own-render`` (we synthesised the audio, so provenance is clean):

  * ``say:<VOICE>@pbas<N>`` — each syllable of a small wordlist spoken by macOS ``say`` voices
    at two base pitches. ``say`` is a system binary; unavailable voices are skipped, not fatal.
  * ``fof:<PRESET>`` — SuperCollider FOF/Chant vowels (a / i / impossible) rendered offline via
    the ``smpl_synth.backends`` NRT bridge + the shipped ``voxFof`` SynthDef, when SC is present.

Auto-alignment of external material is a SEPARATE later task; v0 never ingests anything we
did not render ourselves.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from . import bank as B

# The v0 wordlist — short common words with varied vowel nuclei and onsets, so the seeded
# bank covers the lookup ladder's tiers. Replace per project via build_seed(wordlist=...) or
# the CLI's --wordlist FILE.
WORDLIST = [
    "machine", "sediment", "salt", "water", "signal", "static", "black", "wire", "code",
    "deck", "grid", "tooth", "low", "light", "name", "hum", "rise", "stone", "river", "sing",
    "spit", "kick", "cut", "punch", "clock", "tape", "back", "talk", "wait",
]

SAY_VOICES = ["Fred", "Alex", "Daniel", "Samantha"]
SAY_PBAS = [70, 110]

# FOF vowel presets: (freq, formant table). /a/ = father defaults; /i/ = beet; "impossible" =
# formants stacked above any human vocal tract (the otherworldliness dial).
FOF_PRESETS = {
    "a": {"freq": 110, "phones": ["AA1"],
          "fmt": {"f1": 730, "f2": 1090, "f3": 2440, "f4": 3400, "f5": 4500}},
    "i": {"freq": 165, "phones": ["IY1"],
          "fmt": {"f1": 270, "f2": 2290, "f3": 3010, "f4": 3400, "f5": 4500}},
    "impossible": {"freq": 220, "phones": None,
                   "fmt": {"f1": 1200, "f2": 2800, "f3": 5000, "f4": 7000, "f5": 9000}},
}

def _fof_scd() -> Path | None:
    """The shipped voxFof synthdef, resolved from vox-core package data."""
    try:
        from vox_core import synthdef_path
        return synthdef_path("voxFof")
    except Exception:  # noqa: BLE001 — vox-core absent → FOF source unavailable
        return None


# ---------------------------------------------------------------------------
# DSP helpers (numpy, pure-Python).
# ---------------------------------------------------------------------------
def _trim_silence(x, sr, thresh_frac: float = 0.02, pad_ms: float = 10.0):
    """Trim leading/trailing near-silence by a relative amplitude gate. Keeps a small pad."""
    import numpy as np

    x = np.asarray(x, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0.0:
        return x
    above = np.abs(x) > thresh_frac * peak
    if not above.any():
        return x
    idx = np.nonzero(above)[0]
    pad = int(sr * pad_ms / 1000.0)
    lo = max(0, idx[0] - pad)
    hi = min(len(x), idx[-1] + pad + 1)
    return x[lo:hi]


def _peak_normalize(x, target: float = 0.9):
    import numpy as np

    x = np.asarray(x, dtype="float64")
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0.0:
        return x
    return x * (target / peak)


def _measure_f0(x, sr):
    """Median voiced F0 (Hz) via pyworld harvest+stonemask, or ``None`` if unvoiced."""
    import numpy as np
    import pyworld as pw

    xf = np.ascontiguousarray(np.asarray(x, dtype="float64"))
    if xf.ndim > 1:
        xf = xf.mean(axis=1)
    if xf.size < sr // 20:  # too short to track
        return None
    f0, t = pw.harvest(xf, sr, f0_floor=65.0, f0_ceil=1000.0, frame_period=5.0)
    f0 = pw.stonemask(xf, f0, t, sr)
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return None
    return round(float(np.median(voiced)), 2)


# ---------------------------------------------------------------------------
# say voice source.
# ---------------------------------------------------------------------------
def _available_say_voices(candidates):
    """Filter ``candidates`` to voices actually installed (``say -v '?'``)."""
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    installed = set()
    for line in listing.stdout.splitlines():
        name = line.split("  ")[0].strip()
        if name:
            installed.add(name)
    return [v for v in candidates if v in installed]


def _ortho_syllables(word: str, n: int) -> list[str]:
    """Split ``word`` orthographically into ~``n`` chunks at vowel-run midpoints.

    Approximate text for `say` to voice — the phones stored on the entry are the ground truth;
    this only shapes how each syllable *sounds* when rendered. Aligned to the phonetic syllable
    count by index-clamping in the caller.
    """
    import re

    w = word.lower()
    if n <= 1 or len(w) < 2:
        return [word]
    runs = [(m.start(), m.end()) for m in re.finditer(r"[aeiouy]+", w)]
    if len(runs) < 2:
        return [word]
    cuts = []
    for k in range(len(runs) - 1):
        end_v, start_next = runs[k][1], runs[k + 1][0]
        cuts.append(max(end_v, (end_v + start_next) // 2))
    chunks, prev = [], 0
    for c in cuts:
        chunks.append(w[prev:c])
        prev = c
    chunks.append(w[prev:])
    return [c for c in chunks if c] or [word]


def _say_render(voice: str, pbas: int, text: str, sr: int):
    """Render ``text`` with ``say`` at base pitch ``pbas``; return mono float64 samples or None."""
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="syllabank-say-") as tmp:
        out = str(Path(tmp) / "s.wav")
        try:
            proc = subprocess.run(
                ["say", "-v", voice, "--file-format=WAVE",
                 f"--data-format=LEI16@{sr}", "-o", out, f"[[pbas {pbas}]]{text}"],
                capture_output=True, timeout=30, check=False)
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        p = Path(out)
        if proc.returncode != 0 or not p.exists() or p.stat().st_size == 0:
            return None
        data, _sr = sf.read(out, dtype="float64", always_2d=True)
        return data.mean(axis=1)


# ---------------------------------------------------------------------------
# FOF (SuperCollider) source.
# ---------------------------------------------------------------------------
def _fof_render(preset: dict, dur: float, sr: int):
    """Render one FOF vowel offline via the smpl-synth NRT bridge; return mono float64 or None."""
    import io

    try:
        from smpl_synth import backends
    except Exception:  # noqa: BLE001 — smpl-synth not installed → FOF source unavailable
        return None
    scd = _fof_scd()
    if not backends.sc_available() or scd is None or not scd.exists():
        return None
    import soundfile as sf

    params = {"freq": float(preset["freq"]), "amp": 0.3, "dur": float(dur)}
    params.update({k: float(v) for k, v in preset["fmt"].items()})
    try:
        wav = backends.render_nrt(
            synthdef_source=scd.read_text(), synth_name="voxFof",
            params=params, duration=dur, sr=sr, timeout=60.0)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"  fof render failed: {exc}\n")
        return None
    data, _sr = sf.read(io.BytesIO(wav), dtype="float64", always_2d=True)
    return data.mean(axis=1)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def build_seed(root=None, sr: int = 44100, wordlist=None, voices=None) -> dict:
    """Build the seed bank into ``root``. Returns ``{"entries": N, "by_source": {...}}``."""
    words = list(WORDLIST if wordlist is None else wordlist)
    voice_candidates = list(SAY_VOICES if voices is None else voices)
    voices_ok = _available_say_voices(voice_candidates)
    skipped = [v for v in voice_candidates if v not in voices_ok]
    if skipped:
        sys.stderr.write(f"seed: skipping unavailable say voices: {', '.join(skipped)}\n")

    counts: dict[str, int] = {}
    total = 0

    for word in words:
        phones, syllables = B.word_syllables(word)
        ortho = _ortho_syllables(word, len(syllables))
        for si, syl_phones in enumerate(syllables):
            text = ortho[min(si, len(ortho) - 1)]
            syl_text = text
            for voice in voices_ok:
                for pbas in SAY_PBAS:
                    samples = _say_render(voice, pbas, text, sr)
                    if samples is None:
                        continue
                    samples = _peak_normalize(_trim_silence(samples, sr), 0.9)
                    if len(samples) < sr // 100:  # <10ms of audio → drop
                        continue
                    source = f"say:{voice}@pbas{pbas}"
                    entry_id = f"say-{word}-s{si}-{voice.lower()}-p{pbas}"
                    B.add_entry(root, samples, sr, {
                        "id": entry_id,
                        "syllable": syl_text,
                        "phones": (syl_phones if syl_phones else None),
                        "f0_hz": _measure_f0(samples, sr),
                        "source": source,
                        "license": "own-render",
                        "attribution": None,
                    })
                    counts[source] = counts.get(source, 0) + 1
                    total += 1

    # FOF vowels (SuperCollider), when available.
    for name, preset in FOF_PRESETS.items():
        samples = _fof_render(preset, dur=1.0, sr=sr)
        if samples is None:
            continue
        samples = _peak_normalize(_trim_silence(samples, sr), 0.9)
        source = f"fof:{name}"
        B.add_entry(root, samples, sr, {
            "id": f"fof-{name}",
            "syllable": name,
            "phones": preset["phones"],
            "f0_hz": _measure_f0(samples, sr) or float(preset["freq"]),
            "source": source,
            "license": "own-render",
            "attribution": None,
        })
        counts[source] = counts.get(source, 0) + 1
        total += 1

    sys.stderr.write("seed: per-source entry counts:\n")
    for src in sorted(counts):
        sys.stderr.write(f"  {src}: {counts[src]}\n")
    sys.stderr.write(f"seed: {total} entries total\n")
    return {"entries": total, "by_source": counts}
