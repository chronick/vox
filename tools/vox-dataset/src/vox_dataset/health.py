"""Per-clip and dataset-level health measurements.

Per clip (:func:`measure_clip`): format conformance (sample rate / channels / bit depth),
duration, peak + clipping check, noise-floor estimate (dBFS of the quietest 200 ms), an SNR proxy,
a dryness proxy (reverb-tail energy after speech offsets), bass-safe guarded F0 stats (median /
range / voiced fraction) and — when a transcript or whisper is available — syllable rate and an
ARPABET phone histogram.

Aggregate (:func:`aggregate`, :func:`measure_dataset`): total + usable minutes, a semitone
pitch-coverage histogram, phone coverage vs the full ARPABET inventory, duration distribution, and
dryness / noise / SNR distributions. Everything returned is JSON-safe native Python (no numpy
scalars) — numpy scalars silently break ``json.dumps`` downstream.

F0 is measured with a low-floor pyworld pass (bass-safe) and the trusted median is taken from
smpl-take's shared :func:`measure_f0_guarded` ruler when its src is importable, falling back to the
pyworld voiced median otherwise. faster-whisper is imported lazily and is optional.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import phones

AUDIO_EXTS = (".wav", ".flac", ".aif", ".aiff", ".ogg")

# pyworld low-floor pass: floor low enough to see sub-bass carriers, ceil wide enough for high
# female singing without letting the pass lock onto a partial for most speech.
_F0_FLOOR = 40.0
_F0_CEIL = 1000.0

_SUBTYPE_BITDEPTH = {
    "PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
    "FLOAT": 32, "DOUBLE": 64, "ALAW": 8, "ULAW": 8, "VORBIS": None,
}

# A clipped sample sits at/above this absolute amplitude (full-scale float).
_CLIP_LEVEL = 0.9995
# A clip is flagged as "clipping" once this fraction of its samples are at full scale.
_CLIP_FRAC_FLAG = 1e-4


# --------------------------------------------------------------------------- helpers
def _dbfs(rms: float) -> float | None:
    """dBFS of a linear RMS/amplitude value (None for silence)."""
    if rms is None or rms <= 0:
        return None
    return 20.0 * math.log10(rms)


def _f(v, ndigits: int = 3):
    """Round a nullable numeric to a NATIVE python float (never a numpy scalar)."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return round(float(v), ndigits)


def _bitdepth(subtype: str | None):
    if not subtype:
        return None
    return _SUBTYPE_BITDEPTH.get(subtype.upper())


def _frame_rms(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """RMS per frame (linear). Returns [] for signals shorter than one window."""
    if len(x) < win:
        return np.asarray([np.sqrt(np.mean(x**2))]) if len(x) else np.asarray([])
    idx = np.arange(0, len(x) - win + 1, hop)
    out = np.empty(len(idx), dtype="float64")
    for i, s in enumerate(idx):
        seg = x[s:s + win]
        out[i] = math.sqrt(float(np.mean(seg * seg)))
    return out


def _guarded_median(x: np.ndarray, sr: int, hint: float | None):
    """Trusted bass-safe median F0 via vox-core's shared ruler; None if it can't run."""
    try:
        from vox_core import measure_f0_guarded  # declared dependency

        return measure_f0_guarded(x, int(sr), target_hint=hint).get("f0_hz")
    except Exception:  # noqa: BLE001 — ruler not importable (standalone/test): caller falls back
        return None


def _f0_contour(x: np.ndarray, sr: int):
    """Voiced F0 values (Hz) via a low-floor pyworld harvest+stonemask pass, and the voiced
    fraction over all analysis frames. Returns ``(voiced_hz_array, voiced_frac)``."""
    import pyworld as pw

    xf = np.ascontiguousarray(x, dtype="float64")
    f0, t = pw.harvest(xf, int(sr), f0_floor=_F0_FLOOR, f0_ceil=_F0_CEIL, frame_period=5.0)
    f0 = pw.stonemask(xf, f0, t, int(sr))
    voiced = f0[f0 > 0]
    frac = float(voiced.size) / float(f0.size) if f0.size else 0.0
    return voiced, frac


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0)


# --------------------------------------------------------------------------- per-clip
def measure_clip(
    path: str | Path,
    text: str | None = None,
    whisper_model=None,
    f0_hint: float | None = None,
) -> dict:
    """Full health measurement for one audio file.

    ``text`` is a known transcript (skips whisper). ``whisper_model`` is a loaded
    ``faster_whisper.WhisperModel`` used to transcribe when ``text`` is None. ``f0_hint`` is an
    optional authored/target pitch that keeps the guarded ruler from octave-jumping on bass.
    """
    import soundfile as sf

    path = Path(path)
    info = sf.info(str(path))
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    channels = int(data.shape[1])
    mono = data.mean(axis=1) if channels > 1 else data[:, 0]
    n = int(mono.shape[0])
    dur = n / float(sr) if sr else 0.0

    # Peak + clipping.
    peak = float(np.max(np.abs(mono))) if n else 0.0
    clipped_samples = int(np.count_nonzero(np.abs(mono) >= _CLIP_LEVEL)) if n else 0
    clipped_frac = clipped_samples / n if n else 0.0
    is_clipped = clipped_frac >= _CLIP_FRAC_FLAG

    # Noise floor: quietest 200 ms window (RMS -> dBFS). SNR proxy: loudest 200 ms over noise floor.
    win = max(1, int(0.2 * sr))
    hop = max(1, win // 2)
    rms = _frame_rms(mono, win, hop)
    if rms.size:
        # Clamp the floor RMS to a tiny epsilon so a digitally-silent quietest window reads a very
        # low dBFS (perfectly clean) instead of None — otherwise clean clips lose their noise floor.
        noise_floor = _dbfs(max(float(np.min(rms)), 1e-7))
        signal_level = _dbfs(float(np.percentile(rms, 95)))
    else:
        noise_floor = signal_level = None
    snr = (signal_level - noise_floor) if (signal_level is not None and noise_floor is not None) else None

    # Dryness: reverb-tail proxy — median post-offset / pre-offset energy across speech offsets.
    dryness = _dryness_db(mono, sr, noise_floor)

    # Guarded F0 stats (bass-safe). Contour drives range/voiced-frac; median from the shared ruler.
    voiced, voiced_frac = _f0_contour(mono, sr)
    if voiced.size:
        p5 = float(np.percentile(voiced, 5))
        p95 = float(np.percentile(voiced, 95))
        vmin, vmax = float(np.min(voiced)), float(np.max(voiced))
        contour_median = float(np.median(voiced))
        rng_st = 12.0 * math.log2(p95 / p5) if p5 > 0 else None
    else:
        p5 = p95 = vmin = vmax = contour_median = rng_st = None
    guarded = _guarded_median(mono, sr, f0_hint)
    f0_median = guarded if guarded is not None else contour_median

    # Transcript-driven phone/syllable metrics (optional).
    if text is None and whisper_model is not None:
        text = _whisper_text(mono, sr, whisper_model)
    tstats = phones.transcript_stats(text) if text else None
    voiced_time = voiced_frac * dur
    syllable_rate = None
    if tstats and voiced_time > 0 and tstats["syllables"] > 0:
        syllable_rate = tstats["syllables"] / voiced_time

    return {
        "path": str(path),
        "filename": path.name,
        "sr": int(sr),
        "channels": channels,
        "subtype": info.subtype,
        "bitdepth": _bitdepth(info.subtype),
        "duration_s": _f(dur, 3),
        "peak": _f(peak, 4),
        "peak_dbfs": _f(_dbfs(peak), 2),
        "clipped_samples": clipped_samples,
        "clipped_frac": _f(clipped_frac, 6),
        "is_clipped": bool(is_clipped),
        "noise_floor_dbfs": _f(noise_floor, 2),
        "signal_dbfs": _f(signal_level, 2),
        "snr_db": _f(snr, 2),
        "dryness_db": _f(dryness, 2),
        "f0_median_hz": _f(f0_median, 2),
        "f0_p5_hz": _f(p5, 2),
        "f0_p95_hz": _f(p95, 2),
        "f0_min_hz": _f(vmin, 2),
        "f0_max_hz": _f(vmax, 2),
        "f0_range_semitones": _f(rng_st, 2),
        "voiced_frac": _f(voiced_frac, 4),
        "transcript": text,
        "has_transcript": text is not None,
        "syllables": (tstats or {}).get("syllables"),
        "syllable_rate_hz": _f(syllable_rate, 2),
        "phone_counts": (tstats or {}).get("phone_counts"),
    }


def _dryness_db(x: np.ndarray, sr: int, noise_floor_dbfs: float | None) -> float | None:
    """Reverb-tail proxy: median ratio (dB) of energy in the 150 ms AFTER each speech offset to
    the energy just before it. A dry clip's tail collapses to the noise floor (very negative dB);
    a wet clip keeps energy ringing in the post-offset window (closer to 0 dB). None when no clear
    offsets are found. This is the "energy after offset" idea, made robust by taking the median
    over every offset rather than only the final one (curated clips often have trailing silence)."""
    if len(x) < int(0.2 * sr):
        return None
    fw = max(1, int(0.02 * sr))          # 20 ms energy frames
    env = _frame_rms(x, fw, fw)          # non-overlapping
    if env.size < 6:
        return None
    peak = float(np.max(env))
    if peak <= 0:
        return None
    nf_lin = 10 ** (noise_floor_dbfs / 20.0) if noise_floor_dbfs is not None else peak * 0.01
    gate = max(nf_lin * 3.0, peak * 0.05)
    active = env > gate
    post_win = max(1, int(0.15 * sr / fw))   # 150 ms expressed in frames
    pre_win = post_win
    ratios: list[float] = []
    for i in range(1, len(active) - 1):
        if active[i] and not active[i + 1]:  # offset between frame i and i+1
            pre = env[max(0, i - pre_win + 1): i + 1]
            post = env[i + 1: i + 1 + post_win]
            if pre.size == 0 or post.size == 0:
                continue
            pre_e = float(np.mean(pre))
            post_e = float(np.mean(post))
            if pre_e <= 0:
                continue
            ratios.append(20.0 * math.log10(max(post_e, 1e-9) / pre_e))
    if not ratios:
        return None
    return float(np.median(ratios))


def _whisper_text(x: np.ndarray, sr: int, model) -> str | None:
    """Transcribe with a loaded faster-whisper model (resampled to 16 kHz mono float32)."""
    try:
        from scipy.signal import resample_poly

        xf = np.ascontiguousarray(x, dtype="float64")
        if sr != 16000:
            g = math.gcd(int(sr), 16000)
            xf = resample_poly(xf, 16000 // g, int(sr) // g)
        x16 = xf.astype("float32")
        segments, _info = model.transcribe(x16, language="en", beam_size=1, vad_filter=False)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text or None
    except Exception:  # noqa: BLE001 — whisper is best-effort; a failure means "no transcript"
        return None


def load_whisper(model_size: str = "base"):
    """Load a faster-whisper model (int8/CPU), or return None if the package is absent."""
    try:
        from faster_whisper import WhisperModel

        return WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception:  # noqa: BLE001 — optional dependency
        return None


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- aggregate
def _dist(values: list[float]) -> dict:
    """min/max/mean/median/p10/p90 for a list of numbers (Nones dropped)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None,
                "p10": None, "p90": None}
    a = np.asarray(vals, dtype="float64")
    return {
        "n": int(a.size),
        "min": _f(float(np.min(a)), 3),
        "max": _f(float(np.max(a)), 3),
        "mean": _f(float(np.mean(a)), 3),
        "median": _f(float(np.median(a)), 3),
        "p10": _f(float(np.percentile(a, 10)), 3),
        "p90": _f(float(np.percentile(a, 90)), 3),
    }


def _counts(values) -> dict:
    out: dict = {}
    for v in values:
        key = v
        out[key] = out.get(key, 0) + 1
    # JSON needs string keys for ints too when nested; keep native but stringify for safety.
    return {str(k): int(n) for k, n in sorted(out.items(), key=lambda kv: (kv[0] is None, kv[0]))}


def aggregate(clips: list[dict], usable_predicate=None) -> dict:
    """Roll per-clip measurements up into a dataset report.

    ``usable_predicate`` decides which clips count toward *usable* minutes (default: not clipped
    and duration between 0.3 s and 60 s). The returned dict carries flat scalar metrics (consumed
    by the rubric engine), per-clip value lists (under ``lists``), and distributions.
    """
    if usable_predicate is None:
        def usable_predicate(c):
            d = c.get("duration_s") or 0.0
            return (not c.get("is_clipped")) and 0.3 <= d <= 60.0

    n = len(clips)
    durations = [c["duration_s"] for c in clips if c.get("duration_s") is not None]
    total_minutes = sum(durations) / 60.0
    usable = [c for c in clips if usable_predicate(c)]
    usable_minutes = sum((c.get("duration_s") or 0.0) for c in usable) / 60.0

    # Format distributions.
    sr_counts = _counts([c.get("sr") for c in clips])
    channel_counts = _counts([c.get("channels") for c in clips])
    bitdepth_counts = _counts([c.get("bitdepth") for c in clips])
    subtype_counts = _counts([c.get("subtype") for c in clips])

    def frac(pred) -> float:
        return (sum(1 for c in clips if pred(c)) / n) if n else 0.0

    frac_mono = frac(lambda c: c.get("channels") == 1)
    frac_clipped = frac(lambda c: c.get("is_clipped"))

    # Pitch-coverage histogram (1-semitone MIDI bins) pooled across all voiced frames.
    # Per-clip we only kept summary f0s, so approximate the pool from each clip's [p5, p95] span —
    # every integer semitone the clip demonstrably reaches gets a hit. This gives a coverage map
    # (which notes ARE reached) rather than a density, which is what a bank cares about.
    pitch_hist: dict[int, int] = {}
    reached_midi: list[int] = []
    for c in clips:
        lo, hi = c.get("f0_p5_hz"), c.get("f0_p95_hz")
        if lo and hi and lo > 0 and hi > 0:
            m_lo = int(round(hz_to_midi(lo)))
            m_hi = int(round(hz_to_midi(hi)))
            for m in range(min(m_lo, m_hi), max(m_lo, m_hi) + 1):
                pitch_hist[m] = pitch_hist.get(m, 0) + 1
                reached_midi.append(m)
    if reached_midi:
        pm_lo, pm_hi = min(reached_midi), max(reached_midi)
        pitch_span_semitones = pm_hi - pm_lo
        pitch_lo_hz = 440.0 * 2 ** ((pm_lo - 69) / 12.0)
        pitch_hi_hz = 440.0 * 2 ** ((pm_hi - 69) / 12.0)
    else:
        pm_lo = pm_hi = None
        pitch_span_semitones = 0
        pitch_lo_hz = pitch_hi_hz = None

    # Phone coverage (only meaningful when at least one clip had a transcript).
    have_any_transcript = any(c.get("has_transcript") for c in clips)
    phone_counts: dict[str, int] = {}
    for c in clips:
        for ph, cnt in (c.get("phone_counts") or {}).items():
            phone_counts[ph] = phone_counts.get(ph, 0) + int(cnt)
    if have_any_transcript:
        cov = phones.coverage(phone_counts)
        phone_coverage_weighted_pct = cov["weighted_pct"]
        phone_coverage_raw_pct = cov["raw_pct"]
    else:
        cov = None
        phone_coverage_weighted_pct = None
        phone_coverage_raw_pct = None

    dryness_vals = [c.get("dryness_db") for c in clips]
    noise_vals = [c.get("noise_floor_dbfs") for c in clips]
    snr_vals = [c.get("snr_db") for c in clips]
    peak_vals = [c.get("peak_dbfs") for c in clips]
    syl_rate_vals = [c.get("syllable_rate_hz") for c in clips]

    metrics = {
        "clip_count": n,
        "usable_clip_count": len(usable),
        "total_minutes": _f(total_minutes, 2),
        "usable_minutes": _f(usable_minutes, 2),
        "frac_mono": _f(frac_mono, 4),
        "frac_clipped": _f(frac_clipped, 4),
        "frac_no_clip": _f(1.0 - frac_clipped, 4),
        "pitch_span_semitones": int(pitch_span_semitones),
        "pitch_lo_hz": _f(pitch_lo_hz, 2),
        "pitch_hi_hz": _f(pitch_hi_hz, 2),
        "phone_coverage_weighted_pct": phone_coverage_weighted_pct,
        "phone_coverage_raw_pct": phone_coverage_raw_pct,
        "mean_snr_db": _dist(snr_vals)["mean"],
        "median_noise_floor_dbfs": _dist(noise_vals)["median"],
        "median_dryness_db": _dist(dryness_vals)["median"],
    }

    return {
        "clip_count": n,
        "metrics": metrics,
        "lists": {
            "durations": durations,
            "dryness_db": [v for v in dryness_vals if v is not None],
            "noise_floor_dbfs": [v for v in noise_vals if v is not None],
            "snr_db": [v for v in snr_vals if v is not None],
            "peak_dbfs": [v for v in peak_vals if v is not None],
            "syllable_rate_hz": [v for v in syl_rate_vals if v is not None],
        },
        "format": {
            "sr_counts": sr_counts,
            "channel_counts": channel_counts,
            "bitdepth_counts": bitdepth_counts,
            "subtype_counts": subtype_counts,
        },
        "pitch": {
            "have_data": bool(reached_midi),
            "midi_lo": pm_lo,
            "midi_hi": pm_hi,
            "span_semitones": int(pitch_span_semitones),
            "lo_hz": _f(pitch_lo_hz, 2),
            "hi_hz": _f(pitch_hi_hz, 2),
            "histogram_midi": {str(k): int(v) for k, v in sorted(pitch_hist.items())},
        },
        "phones": {
            "have_transcripts": have_any_transcript,
            "counts": dict(sorted(phone_counts.items())),
            "coverage": cov,
        },
        "distributions": {
            "duration_s": _dist(durations),
            "dryness_db": _dist(dryness_vals),
            "noise_floor_dbfs": _dist(noise_vals),
            "snr_db": _dist(snr_vals),
            "peak_dbfs": _dist(peak_vals),
            "syllable_rate_hz": _dist(syl_rate_vals),
        },
    }


def find_clips(directory: str | Path) -> list[Path]:
    """All audio files under ``directory`` (recursive), sorted."""
    directory = Path(directory)
    out: list[Path] = []
    for ext in AUDIO_EXTS:
        out.extend(directory.rglob(f"*{ext}"))
    return sorted(set(out))


def load_transcripts(csv_path: str | Path) -> dict[str, str]:
    """Load a transcript CSV. Maps filename (with or without extension, basename) -> text.

    Accepts ``filename,text`` rows, with or without a header. Extra columns after the first are
    joined back with commas (so unquoted transcript commas survive)."""
    import csv

    out: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    start = 0
    if rows and rows[0] and rows[0][0].strip().lower() in ("filename", "file", "path", "name"):
        start = 1
    for row in rows[start:]:
        if len(row) < 2:
            continue
        key = Path(row[0].strip()).name
        text = ",".join(row[1:]).strip()
        out[key] = text
        out[Path(key).stem] = text  # allow lookup by stem too
    return out


def measure_dataset(
    directory: str | Path,
    transcripts: dict[str, str] | None = None,
    use_whisper: bool = False,
    whisper_size: str = "base",
    usable_predicate=None,
    progress=None,
) -> dict:
    """Measure every clip under ``directory`` and aggregate. ``transcripts`` maps
    filename/stem -> text; ``use_whisper`` transcribes the rest if faster-whisper is installed."""
    clip_paths = find_clips(directory)
    model = load_whisper(whisper_size) if use_whisper else None
    clips: list[dict] = []
    for i, p in enumerate(clip_paths):
        text = None
        if transcripts:
            text = transcripts.get(p.name) or transcripts.get(p.stem)
        clips.append(measure_clip(p, text=text, whisper_model=model if text is None else None))
        if progress:
            progress(i + 1, len(clip_paths), p)
    report = aggregate(clips, usable_predicate=usable_predicate)
    report["directory"] = str(Path(directory))
    report["clips"] = clips
    report["whisper_used"] = model is not None
    return report
