"""Guarded-F0 tests — self-contained synthetic signals.

    cd packages/vox-core && uv run pytest -q
"""

from __future__ import annotations

import numpy as np
from vox_core import measure_f0_guarded

SR = 44_100


def test_guard_forces_pyworld_on_harsh_bass():
    """A 55 Hz sum-of-40-harmonics is the parselmouth trap (harmonic locking reads it HIGH).

    measure_f0_guarded must come back in the bass band (< 90 Hz): it always runs the low-floor
    pyworld pass and refuses to trust a high parselmouth read on a sub-90 target.
    """
    t = np.arange(int(SR * 2.0)) / SR
    x = np.zeros_like(t)
    for k in range(1, 41):  # 40 harmonics of 55 Hz — harsh, bass, harmonic-rich
        x += np.sin(2 * np.pi * 55.0 * k * t) / k
    x = np.ascontiguousarray(x / np.max(np.abs(x)) * 0.9, dtype="float64")

    # With the authored bass target, the guard forces the low-floor result outright.
    guard = measure_f0_guarded(x, SR, target_hint=55.0)
    assert guard["f0_hz"] is not None and guard["f0_hz"] < 90.0, guard
    assert guard["f0_pyworld_hz"] is not None and guard["f0_pyworld_hz"] < 90.0, guard

    # Even with NO hint, the pyworld pass keeps it in-band (never octave-jumps to a partial).
    guard_nohint = measure_f0_guarded(x, SR)
    assert guard_nohint["f0_hz"] is not None and guard_nohint["f0_hz"] < 90.0, guard_nohint


def test_guard_widens_ceil_for_high_target():
    """A genuinely high (220 Hz) target isn't clipped to None by the 200 Hz bass ceil."""
    t = np.arange(int(SR * 2.0)) / SR
    y = np.zeros_like(t)
    for k in range(1, 6):
        y += np.sin(2 * np.pi * 220.0 * k * t) / k
    y = np.ascontiguousarray(y / np.max(np.abs(y)) * 0.9, dtype="float64")
    guard = measure_f0_guarded(y, SR, target_hint=220.0)
    assert guard["f0_hz"] is not None and abs(guard["f0_hz"] - 220.0) < 10.0, guard
