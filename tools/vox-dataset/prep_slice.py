#!/usr/bin/env python3
"""prep_slice.py — slice source wavs into voicebank-ready segments (5–15 s).

Silence-aware windowing for voicebank dataset prep:
  * clips already in [min, max] s pass through unchanged,
  * longer clips split into ~target-length windows, snapping each cut to the
    lowest-energy point within ±1 s (clean phrase/breath boundaries),
  * clips shorter than `min` are dropped (counted).
Mono in, mono out; sample rate preserved. Source-agnostic (recurses `--src`).

    prep_slice.py --src <dir> --dst <dir> [--target 10 --min 5 --max 15]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME_S = 0.02  # 20 ms RMS frames for cut-point search


def _rms(x: np.ndarray, sr: int):
    n = max(1, int(sr * FRAME_S))
    pad = (-len(x)) % n
    fr = np.pad(x, (0, pad)).reshape(-1, n)
    return np.sqrt((fr ** 2).mean(axis=1) + 1e-12), n


def _snap_cut(x: np.ndarray, sr: int, lo: int, hi: int) -> int:
    """Lowest-energy sample index in [lo, hi] — cut where the voice is quietest."""
    if hi <= lo:
        return (lo + hi) // 2
    r, n = _rms(x[lo:hi], sr)
    return lo + int(np.argmin(r)) * n


def slice_file(path: Path, dst: Path, target: float, mn: float, mx: float):
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)  # downmix to mono
    dur = len(x) / sr
    if dur < mn:
        return 0, 1  # too short → drop
    if dur <= mx:
        segs = [(0, len(x))]
    else:
        nwin = max(1, round(dur / target))
        step = len(x) / nwin
        bounds = [0]
        for k in range(1, nwin):
            c = int(k * step)
            lo = max(bounds[-1] + int(mn * sr), c - sr)
            hi = min(len(x) - int(mn * sr), c + sr)
            bounds.append(_snap_cut(x, sr, lo, hi))
        bounds.append(len(x))
        segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    stem = path.stem
    written = 0
    for i, (a, b) in enumerate(segs):
        seg = x[a:b]
        if len(seg) / sr < mn:
            continue
        sf.write(str(dst / f"{stem}__{i:02d}.wav"), seg, sr, subtype="PCM_16")
        written += 1
    return written, 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--target", type=float, default=10.0)
    ap.add_argument("--min", type=float, default=5.0)
    ap.add_argument("--max", type=float, default=15.0)
    a = ap.parse_args()
    src, dst = Path(a.src).expanduser(), Path(a.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    wavs = sorted(src.rglob("*.wav"))
    tot_w = tot_d = 0
    for w in wavs:
        wr, dr = slice_file(w, dst, a.target, a.min, a.max)
        tot_w += wr
        tot_d += dr
    print(f"sliced {len(wavs)} files -> {tot_w} segments ({tot_d} dropped < {a.min}s) -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
