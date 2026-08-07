"""Corpus ingest gate — peak-normalize + VAD-survivability. Pure numpy, no smplstream import.

The admission contract for a voice-conversion matching corpus (e.g. kNN-VC). Two failures a raw
clip can carry into the matching set:

  1. **Level** — a quiet clip contributes weak keys and skews normalization. Fix: peak-normalize
     to ``target_peak`` (0.9) on admission.
  2. **VAD death** — kNN-VC's ``get_matching_set`` runs voice-activity detection; a steady-state
     or near-silent clip (the ``metal-buzz.wav`` failure) gets VADed down to *nothing*, and the
     downstream reshape crashes with "cannot reshape tensor of 0 elements". A VAD keys on energy
     *dynamics* (onsets/offsets), so a flat buzz has nothing to keep.

This module's ``survives`` is a torch-free proxy for (2): a clip survives if its short-time energy
has enough dynamic range (syllabic/gestural modulation) AND isn't essentially silent. It is
deliberately conservative — the authoritative VAD runs in the svc path — but it catches the whole
steady-state failure class before it can poison the set.
"""

from __future__ import annotations

import numpy as np

GATE_OP_VERSION = "corpus-gate@1"

TARGET_PEAK = 0.9
MIN_PEAK_DBFS = -40.0      # quieter than this ≈ silence
MIN_VOICED_FRAC = 0.03     # essentially no above-floor content → nothing for the VAD to keep
# Steady-state (VAD-death) signature: LOW energy modulation AND essentially no gaps. Either
# alone is fine — a dynamic clip may be gapless (legato), a quiet-ish clip may still pulse — but
# a clip that is BOTH flat and gapless is a drone a VAD strips to nothing (the metal-buzz class).
STEADY_MODULATION_DB = 10.0
STEADY_VOICED_FRAC = 0.97


def _ste_db(x, sr, hop_ms=20.0):
    hop = max(int(sr * hop_ms / 1000.0), 1)
    rms = np.array([np.sqrt(np.mean(x[s:s + hop] ** 2) + 1e-12)
                    for s in range(0, max(len(x) - hop, 1), hop)])
    return 20.0 * np.log10(np.maximum(rms, 1e-9))


def peak_normalize(x, target_peak=TARGET_PEAK):
    peak = float(np.max(np.abs(x))) or 1.0
    return (x / peak * target_peak).astype("float32"), peak


def survives(x, sr) -> dict:
    """Torch-free VAD-survivability verdict for one mono signal → a decision dict."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.ascontiguousarray(x, dtype="float64")
    peak = float(np.max(np.abs(x)))
    peak_dbfs = 20.0 * np.log10(peak) if peak > 0 else -120.0

    ste = _ste_db(x, sr)
    ref = float(np.max(ste)) if ste.size else -120.0
    voiced_frac = float(np.mean(ste > ref - 20.0)) if ste.size else 0.0
    modulation = float(np.percentile(ste, 90) - np.percentile(ste, 10)) if ste.size else 0.0

    reasons = []
    if peak_dbfs < MIN_PEAK_DBFS:
        reasons.append(f"too quiet (peak {peak_dbfs:.1f} dBFS < {MIN_PEAK_DBFS})")
    if voiced_frac < MIN_VOICED_FRAC:
        reasons.append(f"almost no above-floor content (voiced_frac {voiced_frac:.3f})")
    if modulation < STEADY_MODULATION_DB and voiced_frac > STEADY_VOICED_FRAC:
        reasons.append(f"steady-state drone — energy modulation {modulation:.1f} dB "
                       f"< {STEADY_MODULATION_DB} with no gaps (voiced_frac {voiced_frac:.3f}); "
                       f"a VAD would strip it to nothing (cf. metal-buzz)")

    return {
        "survives": not reasons,
        "reasons": reasons,
        "peak_dbfs": round(float(peak_dbfs), 2),
        "modulation_db": round(float(modulation), 2),
        "voiced_frac": round(float(voiced_frac), 4),
        "dur_s": round(len(x) / sr, 3),
    }


def gate_file(path, sr_hint=None):
    """Read a WAV, return ``(verdict_dict, normalized_float32, sr)``. Verdict includes level +
    survivability; the normalized signal is what should be written to the matching set on admit."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    verdict = survives(mono, sr)
    normalized, in_peak = peak_normalize(mono, TARGET_PEAK)
    verdict["in_peak"] = round(float(in_peak), 4)
    return verdict, normalized, sr
