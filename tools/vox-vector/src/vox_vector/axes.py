"""Axis mapping — audio + upstream voice descriptors → a six-axis perceptual coordinate.

Each axis is BOTH a control coordinate (0→1) and a *measured* quantity, so a render's
measured position can be diffed against the position a target programmed. That diff is
what makes a rendered take self-verifying: render error becomes a number.

Axis              0 ────────────► 1              measured by (v0 proxy)
  humanness       machine ── human-adjacent      HNR + natural jitter band (coarse; embedding tier pending)
  breathiness     pure tone ── aspirate          HF spectral flatness
  roughness       harmonic ── inharmonic         off-harmonic-grid energy
  intelligibility texture ── intelligible        (ASR word-error — language tier; None in v0)
  multiplicity    solo ── stacked cloud          (voice-count/embedding — embedding tier; None in v0)
  spatiality      dry-close ── vast              late/peak energy-tail sustain (coarse RT proxy)

`humanness`, `intelligibility`, `multiplicity` are explicitly coarse-or-absent in v0 —
honestly flagged in ``proxy_notes`` — because they need embedding/language tiers.
`breathiness`, `roughness`, `spatiality` are real signal measures now.
"""

from __future__ import annotations

import numpy as np

AXES = ("humanness", "breathiness", "roughness", "intelligibility", "multiplicity", "spatiality")


def _clamp(v):
    return float(min(max(v, 0.0), 1.0))


def _spectral_flatness_hf(x, sr, lo_hz=3000.0):
    n = len(x)
    if n < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    hf = spec[freqs >= lo_hz]
    if hf.size < 4:
        return 0.0
    return float(np.exp(np.mean(np.log(hf))) / np.mean(hf))


def _autocorr_f0(x, sr, fmin=65.0, fmax=1000.0):
    """Cheap single-F0 estimate (median over frames) so `roughness` works with no upstream frame."""
    hop = int(sr * 0.02)
    win = int(sr * 0.04)
    ests = []
    for s in range(0, len(x) - win, hop):
        fr = x[s:s + win] * np.hanning(win)
        ac = np.correlate(fr, fr, "full")[win - 1:]
        lo, hi = int(sr / fmax), int(sr / fmin)
        if hi >= len(ac):
            continue
        seg = ac[lo:hi]
        if seg.size and seg.max() > 0.3 * ac[0]:
            ests.append(sr / (lo + int(np.argmax(seg))))
    return float(np.median(ests)) if ests else 0.0


def _inharmonicity(x, sr, f0):
    n = len(x)
    if n < 512 or f0 <= 0:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total = float(np.sum(spec)) + 1e-12
    harm = 0.0
    for k in range(1, int((sr / 2) / f0) + 1):
        harm += float(np.sum(spec[np.abs(freqs - k * f0) <= 0.03 * k * f0]))
    return _clamp(1.0 - harm / total)


def _spatiality(x, sr):
    """Coarse dry↔vast proxy: reverberant energy PERSISTING after the signal's offset.

    Space is only observable where a voice stops but energy continues (a decay tail). So: find
    the last strong-energy frame (>½ peak), then measure how much energy lingers AFTER it. A dry
    close voice tails to silence (→0); a reverberant/vast one holds a tail (→ higher). A clip that
    sustains to its end gives NO offset to measure against → 0.0 with a low-confidence note (you
    can't read room off a held tone) — honest, rather than a false "deep". RT60/DRR on a gated
    tail is the real metric.
    """
    hop = max(int(sr * 0.02), 1)
    rms = np.array([np.sqrt(np.mean(x[s:s + hop] ** 2) + 1e-12) for s in range(0, len(x) - hop, hop)])
    if rms.size < 8:
        return 0.0
    peak = float(rms.max()) or 1.0
    strong = np.where(rms > 0.5 * peak)[0]
    if strong.size == 0:
        return 0.0
    offset = int(strong[-1])
    if offset >= rms.size - 3:               # sustains to the end → no tail to measure
        return 0.0
    tail = rms[offset + 1:]
    return _clamp(float(np.mean(tail) / peak) * 2.0)  # elevated lingering energy = more space


def _humanness(hnr_db, jitter):
    """Coarse machine↔human proxy. Human voice: moderate HNR + a little natural jitter. Pure
    synthesis is either sterile (very high HNR, ~0 jitter) or noisy (very low HNR). Peaks the
    'human' end in the mid-HNR band with non-zero jitter. Embedding-distance is the real metric."""
    if hnr_db is None:
        return None
    # HNR humanness: a broad hump centred ~18 dB, falling off toward sterile (>35) and noisy (<5)
    hnr_hum = np.exp(-((hnr_db - 18.0) ** 2) / (2 * 12.0 ** 2))
    jit_hum = 1.0 if jitter is None else _clamp(jitter / 0.02)  # some jitter reads as human
    return _clamp(0.6 * hnr_hum + 0.4 * jit_hum)


def measure(x, sr, upstream: dict | None = None) -> dict:
    """Map a mono signal (+ optional upstream ``voice.*`` descriptors) onto the six axes.

    ``upstream`` is a flat dict of registered ``voice.*`` keys (from a `vox ear` frame); when
    absent, breathiness/roughness/spatiality are still computed from the audio directly.
    """
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.ascontiguousarray(x, dtype="float64")
    up = upstream or {}

    flatness = up.get("voice.spectral_flatness_hf")
    if flatness is None:
        flatness = _spectral_flatness_hf(x, sr)
    f0 = up.get("voice.f0_median_hz") or _autocorr_f0(x, sr)
    inharm = up.get("voice.inharmonicity")
    if inharm is None:
        inharm = _inharmonicity(x, sr, f0)

    axes = {
        "humanness": _humanness(up.get("voice.hnr_db"), up.get("voice.jitter_local")),
        "breathiness": _clamp(float(flatness) * 6.0),   # flatness_hf ~0.03 tonal … ~0.15 breathy
        "roughness": _clamp(float(inharm)),
        "intelligibility": None,   # language tier (ASR word-error) — not computable in v0
        "multiplicity": None,      # embedding tier (voice-count) — not computable in v0
        "spatiality": _spatiality(x, sr),
    }
    proxy_notes = {
        "humanness": "coarse HNR+jitter heuristic; embedding-distance is the real metric",
        "intelligibility": "needs ASR word-error (language tier)",
        "multiplicity": "needs voice-count/speaker-embedding (embedding tier)",
        "spatiality": "coarse late/peak energy-tail proxy; RT60/DRR is the real metric",
    }
    return {"vector": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in axes.items()},
            "proxy_notes": proxy_notes,
            "f0_median_hz": round(float(f0), 2) if f0 else None}


def diff(programmed: dict, measured: dict) -> dict:
    """Per-axis |programmed − measured| for axes present (non-None) in BOTH, plus the mean.

    This is the self-verifying number: how far a render landed from the coordinate the target
    asked for. Axes that are None on either side are skipped (reported under ``skipped``).
    """
    errs = {}
    skipped = []
    for ax in AXES:
        p, m = programmed.get(ax), measured.get(ax)
        if p is None or m is None:
            skipped.append(ax)
            continue
        errs[ax] = round(abs(float(p) - float(m)), 4)
    mean = round(float(np.mean(list(errs.values()))), 4) if errs else None
    return {"per_axis_error": errs, "mean_abs_error": mean, "skipped": skipped}
