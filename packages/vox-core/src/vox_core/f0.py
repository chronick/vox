"""The bass-safe guarded F0 ruler.

parselmouth's autocorrelation tracker has a 65 Hz floor, so it cannot see deep-bass
material (55–62 Hz) and, worse, its failure mode on harsh/inharmonic bass is to lock
onto a partial or formant and read the pitch HIGH (a 55 Hz FM growl can read ~166 Hz),
silently throwing a bass voice out of its true 45–90 Hz band. The guard here always
runs a low-floor pyworld pass alongside parselmouth and refuses to trust parselmouth
on bass.
"""

from __future__ import annotations

import contextlib

import numpy as np

LOWFLOOR_HZ = 30.0          # pyworld f0_floor for the guarded pass — sees sub-65 Hz voices
LOWFLOOR_CEIL_HZ = 200.0    # default ceil: constrains the bass pass so it can't itself lock high
HIGH_HINT_HZ = 130.0        # above this authored target, widen the ceil so highs aren't clipped
BASS_HINT_HZ = 90.0         # at/below this authored target, force the low-floor result outright
AGREE_CENTS = 100.0         # trust parselmouth only when it agrees with pyworld within this

# parselmouth's native tracking range (kept in sync with vox-ear's descriptors).
_PM_F0_MIN = 65.0
_PM_F0_MAX = 1000.0


def _f0_pyworld(y: np.ndarray, sr: int, floor: float, ceil: float):
    """Median voiced F0 via a pyworld harvest+stonemask pass over ``[floor, ceil]``."""
    import pyworld as pw

    x = np.ascontiguousarray(y, dtype="float64")
    f0, t = pw.harvest(x, sr, f0_floor=float(floor), f0_ceil=float(ceil), frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    voiced = f0[f0 > 0]
    return float(np.median(voiced)) if voiced.size else None


def _f0_parselmouth(y: np.ndarray, sr: int):
    """Median voiced F0 via parselmouth (its native 65 Hz floor) — the value we guard AGAINST."""
    import parselmouth

    x = np.ascontiguousarray(y, dtype="float64")
    snd = parselmouth.Sound(x, sampling_frequency=int(sr))
    pitch = snd.to_pitch(pitch_floor=_PM_F0_MIN, pitch_ceiling=_PM_F0_MAX)
    f = pitch.selected_array["frequency"]
    voiced = f[f > 0]
    return float(np.median(voiced)) if voiced.size else None


def measure_f0_guarded(x: np.ndarray, sr: int, target_hint: float | None = None) -> dict:
    """Bass-safe F0 ruler. Returns ``{f0_hz, f0_source, f0_pyworld_hz, f0_parselmouth_hz}``.

    ALWAYS runs a low-floor pyworld pass (harvest f0_floor=30, stonemask, voiced median)
    alongside parselmouth, then decides which to trust:

    * ``target_hint`` at/below :data:`BASS_HINT_HZ` (90 Hz) → force the pyworld value outright.
      This is the parselmouth trap: on harsh/inharmonic bass parselmouth locks onto a partial
      or formant and reads HIGH, silently ejecting the voice from its 45–90 Hz band.
    * otherwise trust parselmouth ONLY when it agrees with pyworld within :data:`AGREE_CENTS`
      (100 cents); on disagreement (or when parselmouth reads unvoiced) fall back to pyworld.

    The pyworld ceil defaults to :data:`LOWFLOOR_CEIL_HZ` (200 Hz), which keeps the bass pass
    from itself locking onto a high partial; for an authored ``target_hint`` above
    :data:`HIGH_HINT_HZ` (e.g. 180/220 Hz upper voices) the ceil is widened to 1000 Hz so a
    genuinely high fundamental isn't clipped to None. parselmouth itself is optional: when the
    wheel is absent the ruler degrades to pyworld-only.
    """
    ceil = LOWFLOOR_CEIL_HZ
    if target_hint is not None and float(target_hint) >= HIGH_HINT_HZ:
        ceil = 1000.0
    pw_f0 = _f0_pyworld(x, sr, LOWFLOOR_HZ, ceil)

    pm_f0 = None
    with contextlib.suppress(Exception):  # best-effort on odd signals / absent wheel
        pm_f0 = _f0_parselmouth(x, sr)

    if target_hint is not None and float(target_hint) <= BASS_HINT_HZ:
        chosen, source = pw_f0, "pyworld-forced-bass"
    elif pm_f0 is None:
        chosen, source = pw_f0, "pyworld-only"
    elif pw_f0 is None:
        chosen, source = pm_f0, "parselmouth-only"
    else:
        cents = abs(1200.0 * np.log2(pm_f0 / pw_f0))
        if cents <= AGREE_CENTS:
            chosen, source = pm_f0, "parselmouth-agrees"
        else:
            chosen, source = pw_f0, "pyworld-disagree"

    return {
        "f0_hz": chosen,
        "f0_source": source,
        "f0_pyworld_hz": pw_f0,
        "f0_parselmouth_hz": pm_f0,
    }
