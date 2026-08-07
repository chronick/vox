"""concat — the SPLICED-SAMPLE render path: sing lyric lines by stitching real syllable clips
from the license-gated syllabank, WORLD-imposing each syllable's exact per-syllable pitch.

Where ``render.render`` (the say+larynx backend) synthesises every syllable fresh from macOS
``say``, ``render_concat`` looks each syllable up in the syllabank (``vox_syllabank.bank``),
picks the top candidate through the fallback ladder, and pitches THAT real clip to the score's
note. The whole point is provenance: every stitched clip carries a ``source`` + ``license`` from
the bank, and the manifest re-states it per syllable — an un-provenanced sample can never enter
the mix (``load_bank`` already refuses one on read).

Per syllable the path is: bank ``lookup`` (or a ``say`` fallback when phones are null / the bank
is empty) -> ``vox_larynx.world.render`` to impose the exact target Hz (formants kept, no
chipmunking) AND time-stretch to fill the syllable's gated duration (no dead air) -> resample to
the output rate -> equal-power crossfade onto the grid. Siblings ``vox_syllabank`` /
``vox_larynx`` are resolved on PYTHONPATH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .render import note_to_hz_any, say_available
from .schema import load_score

# WORLD time-stretch ceiling — a clip may be stretched up to 4x to fill a long note; beyond that
# the artifacts dominate, so we cap and accept a shorter segment (recorded honestly in the row).
_TIME_RATIO_CAP = 4.0


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample a mono float64 signal ``src_sr`` -> ``dst_sr`` (polyphase; identity when equal)."""
    if src_sr == dst_sr or x.size == 0:
        return x
    from math import gcd

    import scipy.signal as ss

    g = gcd(int(src_sr), int(dst_sr))
    up, down = int(dst_sr) // g, int(src_sr) // g
    return np.asarray(ss.resample_poly(x, up, down), dtype="float64")


def _equal_power_fades(seg: np.ndarray, xfade: int) -> np.ndarray:
    """Apply an equal-power (sin/cos) fade-in and fade-out of ``xfade`` samples to a copy of
    ``seg``. Overlapping a cos fade-out with the next segment's sin fade-in sums to constant
    power (sin^2 + cos^2 = 1) — a click-free crossfade at the join."""
    if xfade <= 1 or seg.size == 0:
        return seg
    seg = seg.copy()
    n = min(xfade, seg.size)
    fin = np.sin(np.linspace(0.0, np.pi / 2.0, n))
    fout = np.cos(np.linspace(0.0, np.pi / 2.0, n))
    seg[:n] *= fin
    seg[-n:] *= fout
    return seg


def _load_clip(bank: dict, entry: dict):
    """Load a bank entry's WAV as ``(mono float64, sr)``."""
    import soundfile as sf

    path = Path(bank["root"]) / entry["file"]
    data, csr = sf.read(str(path), dtype="float64", always_2d=True)
    return data.mean(axis=1), int(csr)


def _say_clip(text: str, voice: str, sr: int):
    """Fallback glottal source: render ``text`` via macOS ``say`` -> ``(mono float64, sr)``."""
    from . import render as render_mod

    return render_mod._say_render(text, voice, sr), sr


def _fit_clip(clip: np.ndarray, csr: int, hz, dur_s: float, out_sr: int, world):
    """Pitch a source clip to ``hz`` (None = leave pitch) and time-stretch to fill ``dur_s`` when
    it is shorter, then resample to ``out_sr``. Returns ``(segment float64, time_ratio)``.

    ``time_ratio`` is the WORLD frame-axis stretch actually applied (1.0 = none), capped at
    ``_TIME_RATIO_CAP`` and recorded in the manifest — the duration-fill audit trail.
    """
    clip = np.asarray(clip, dtype="float64")
    natural_s = len(clip) / float(csr) if csr else 0.0
    time_ratio = 1.0
    if natural_s > 0.0 and dur_s > natural_s:
        time_ratio = min(dur_s / natural_s, _TIME_RATIO_CAP)
    rendered, _params = world.render(clip, csr, to_hz=hz, time_ratio=time_ratio)
    rendered = np.asarray(rendered, dtype="float64")
    rendered = _resample(rendered, csr, out_sr)
    return rendered, float(time_ratio)


def render_concat(score, bank_root=None, sr: int = 44100, crossfade_ms: float = 12.0,
                  voice: str = "Fred"):
    """Render a TONGUE score by STITCHING spliced syllable clips from the syllabank.

    Per syllable: look the phones up in the bank (``vox_syllabank.bank.lookup`` with the note Hz
    as the f0 hint) and take the top candidate; load its WAV; WORLD-impose the exact note Hz AND
    duration-fill (time-stretch, capped) to the gated ``dur_beats``; resample to ``sr``; place on
    the grid with equal-power crossfades. Null/absent phones (or an empty bank) fall back to a
    ``say``-rendered clip for that syllable, flagged in the manifest.

    Returns ``(samples float32 mono, manifest)`` where ``manifest`` is
    ``{"syllables": [row, ...], "provenance": {...}}`` — one row per syllable
    ``{text, phones, bank_id, tier, source, license, t, dur, hz, time_ratio}`` (``bank_id`` is
    the entry id, or the literal ``"say"`` for a fallback). THE MANIFEST IS REQUIRED: it is the
    provenance record every stitched clip must carry.
    """
    from vox_larynx import world  # sibling tool, resolved on PYTHONPATH
    from vox_syllabank.bank import (  # sibling tool, resolved on PYTHONPATH
        load_bank,
        lookup,
    )

    score = load_score(score)
    bpm = float(score["meta"]["bpm"])
    spb = 60.0 / bpm
    bank = load_bank(bank_root)  # re-validates provenance; raises on an un-provenanced entry
    xfade = int(sr * crossfade_ms / 1000.0)

    placed: list[tuple[int, np.ndarray]] = []  # (start_sample, signal)
    rows: list[dict] = []

    for syl in score["syllables"]:
        phones = syl["phones"]
        hz = note_to_hz_any(syl["note"])
        dur_s = float(syl["dur_beats"]) * spb
        target = max(1, int(round(dur_s * sr)))

        entry = None
        tier = None
        if phones:  # non-null, non-empty -> try the bank
            hits = lookup(bank, phones, f0_hint=hz, top=1)
            if hits:
                entry, tier = hits[0]

        if entry is not None:
            clip, csr = _load_clip(bank, entry)
            bank_id = entry["id"]
            source = entry.get("source")
            license_ = entry.get("license")
        else:
            if not say_available():
                raise RuntimeError(
                    f"syllable {syl['text']!r}: no bank match and `say` unavailable for fallback")
            clip, csr = _say_clip(syl["text"], voice, sr)
            bank_id = "say"
            source = f"say:{voice}"
            license_ = "own-render"  # we synthesised it — provenance is clean by construction

        seg, time_ratio = _fit_clip(clip, csr, hz, dur_s, sr, world)
        seg = seg * float(syl.get("dyn", 1.0))
        # Keep the body plus one crossfade tail so it overlaps the next segment's head.
        keep = min(len(seg), target + xfade)
        seg = _equal_power_fades(seg[:keep], xfade)

        start = int(round(float(syl["start_beat"]) * spb * sr))
        placed.append((start, seg))
        rows.append({
            "text": syl["text"],
            "phones": phones,
            "bank_id": bank_id,
            "tier": tier,
            "source": source,
            "license": license_,
            "t": float(syl["start_beat"] * spb),
            "dur": float(dur_s),
            "hz": float(hz) if hz is not None else None,
            "time_ratio": float(time_ratio),
        })

    total = max((start + len(g) for start, g in placed), default=0) + 1
    out = np.zeros(total, dtype="float64")
    for start, g in placed:
        end = min(start + len(g), total)
        out[start:end] += g[: end - start]

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out = out / peak * 0.9

    manifest = {"syllables": rows, "provenance": _provenance(rows)}
    return out.astype("float32"), manifest


def _provenance(rows: list[dict]) -> dict:
    """Top-level provenance summary over the per-syllable rows — the audit at a glance.

    ``unlicensed`` MUST be 0: a row without a license is a provenance leak and the acceptance
    gate fails on it.
    """
    tiers: dict[str, int] = {}
    for r in rows:
        key = "say" if r["tier"] is None else f"tier{r['tier']}"
        tiers[key] = tiers.get(key, 0) + 1
    return {
        "n_syllables": len(rows),
        "n_bank": sum(1 for r in rows if r["bank_id"] != "say"),
        "n_say_fallback": sum(1 for r in rows if r["bank_id"] == "say"),
        "licenses": sorted({str(r["license"]) for r in rows if r["license"]}),
        "sources": sorted({str(r["source"]) for r in rows if r["source"]}),
        "tiers": tiers,
        "unlicensed": sum(1 for r in rows if not r["license"]),
    }
