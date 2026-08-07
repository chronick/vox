"""Voice-native descriptors — pure parselmouth/numpy, no smplstream import.

Returns a flat dict of registered ``voice.*`` feature keys (see the repo's
feature-keys.md). This is the deterministic measurement tier: the things a
listener can hear but not quantify — formant positions, breathiness (HNR),
pitch/amplitude perturbation (jitter/shimmer), vibrato rate/extent — measured
identically on every render so voices can be compared as ranked numbers, not
raw listens. Two signal-level supports ride along: ``spectral_flatness_hf``
(a breathiness proxy) and ``inharmonicity`` (a clangor proxy). ML-heavy
measures (embeddings, transcription, voice counting) belong to other tools by
design.
"""

from __future__ import annotations

import contextlib

import numpy as np

DESCRIBE_OP_VERSION = "ear-describe@1"

F0_MIN = 65.0
F0_MAX = 1000.0


def _pitch_values(snd):
    pitch = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    f = pitch.selected_array["frequency"]
    dt = pitch.time_step
    return f, dt


def _vibrato(f0_voiced_cents: np.ndarray, dt: float) -> tuple:
    """Rate (Hz) + extent (cents) of the dominant 3–8 Hz modulation in the F0 contour.

    Detrends the voiced F0 (already in cents rel. its median), takes the magnitude spectrum,
    and picks the peak inside the vibrato band. Extent is 2× the RMS of the detrended contour
    (a robust peak-to-peak proxy). Too-short/flat contours return (None, None).
    """
    n = f0_voiced_cents.size
    if n < 8 or dt <= 0:
        return None, None
    x = f0_voiced_cents - np.mean(f0_voiced_cents)
    # remove slow drift (linear detrend) so a rising phrase isn't read as huge vibrato
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, dt)
    band = (freqs >= 3.0) & (freqs <= 8.0)
    if not band.any() or spec[band].max() <= 0:
        return None, round(float(2 * np.std(x)), 1)
    rate = float(freqs[band][np.argmax(spec[band])])
    return round(rate, 2), round(float(2 * np.std(x)), 1)


def _spectral_flatness_hf(x: np.ndarray, sr: int, lo_hz: float = 3000.0) -> float:
    """Geometric/arithmetic-mean power ratio above ``lo_hz`` (0=tonal, →1=noise). Breath proxy."""
    n = len(x)
    if n < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    hf = spec[freqs >= lo_hz]
    if hf.size < 4:
        return 0.0
    gmean = np.exp(np.mean(np.log(hf)))
    amean = np.mean(hf)
    return round(float(gmean / amean), 4)


def _inharmonicity(x: np.ndarray, sr: int, f0_hz: float) -> float:
    """Fraction of spectral energy NOT within ±3% of an f0 harmonic (0=pure harmonic, →1=metallic).

    A cheap clangor proxy: sum power in narrow bands around n·f0 and compare to total. An
    inharmonic/clangorous timbre spreads energy off the harmonic grid → higher value.
    """
    n = len(x)
    if n < 512 or not f0_hz or f0_hz <= 0:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total = float(np.sum(spec)) + 1e-12
    harmonic = 0.0
    for k in range(1, int((sr / 2) / f0_hz) + 1):
        fc = k * f0_hz
        mask = np.abs(freqs - fc) <= 0.03 * fc
        harmonic += float(np.sum(spec[mask]))
    return round(float(1.0 - min(harmonic / total, 1.0)), 4)


def describe(x: np.ndarray, sr: int) -> dict:
    """Full voice-descriptor dict (registered ``voice.*`` keys). Unvoiced/short → Nones + note."""
    import parselmouth
    from parselmouth.praat import call

    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.ascontiguousarray(x, dtype="float64")
    snd = parselmouth.Sound(x, sampling_frequency=sr)

    f0, dt = _pitch_values(snd)
    voiced = f0[f0 > 0]
    voiced_frac = round(float((f0 > 0).mean()), 4) if f0.size else 0.0

    out: dict = {
        "voice.voiced_frac": voiced_frac,
        "voice.f0_median_hz": None, "voice.f0_range_semitones": None,
        "voice.f1_hz": None, "voice.f2_hz": None, "voice.f3_hz": None, "voice.f4_hz": None,
        "voice.hnr_db": None, "voice.jitter_local": None, "voice.shimmer_local_db": None,
        "voice.vibrato_rate_hz": None, "voice.vibrato_extent_cents": None,
        "voice.spectral_flatness_hf": _spectral_flatness_hf(x, sr),
        "voice.inharmonicity": 0.0,
    }
    if voiced.size < 3:
        out["note"] = "insufficient voiced frames (unvoiced/too-short): timbre keys null"
        return out

    med = float(np.median(voiced))
    lo, hi = float(np.min(voiced)), float(np.max(voiced))
    out["voice.f0_median_hz"] = round(med, 2)
    out["voice.f0_range_semitones"] = round(float(12.0 * np.log2(hi / lo)), 2) if lo > 0 else None

    # Formants F1–F4 (mean over the whole sound; Burg tracker).
    formant = snd.to_formant_burg(maximum_formant=min(5500.0, 0.49 * sr))
    dur = snd.get_total_duration()
    times = np.linspace(0.1 * dur, 0.9 * dur, 15)
    for i in range(1, 5):
        vals = [formant.get_value_at_time(i, t) for t in times]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        out[f"voice.f{i}_hz"] = round(float(np.mean(vals)), 1) if vals else None

    # HNR (breathiness inverse).
    harm = snd.to_harmonicity_cc()
    hv = harm.values[harm.values != -200]
    out["voice.hnr_db"] = round(float(np.mean(hv)), 2) if hv.size else None

    # Jitter / shimmer via a periodic PointProcess; best-effort on odd signals.
    with contextlib.suppress(Exception):
        pp = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
        jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shim = call([snd, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        out["voice.jitter_local"] = round(float(jit), 5) if not np.isnan(jit) else None
        out["voice.shimmer_local_db"] = round(float(shim), 4) if not np.isnan(shim) else None

    # Vibrato from the voiced F0 contour (in cents rel. median).
    cents = 1200.0 * np.log2(np.clip(voiced, 1e-6, None) / med)
    rate, extent = _vibrato(cents, dt)
    out["voice.vibrato_rate_hz"] = rate
    out["voice.vibrato_extent_cents"] = extent

    out["voice.inharmonicity"] = _inharmonicity(x, sr, med)
    return out
