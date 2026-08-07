"""Per-clip and aggregate measurement tests — conformance/clipping/noise fire correctly, and
all returned values are JSON-safe native python (no numpy scalars)."""

from __future__ import annotations

import json

import numpy as np

from vox_dataset import health


def _no_numpy_scalars(obj):
    """Recursively assert no numpy scalar leaks into the report (they break json.dumps)."""
    if isinstance(obj, dict):
        for v in obj.values():
            _no_numpy_scalars(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _no_numpy_scalars(v)
    else:
        assert not isinstance(obj, np.generic), f"numpy scalar leaked: {obj!r}"


def test_clip_conformance_and_clipping(mini_dataset):
    clip = health.measure_clip(mini_dataset / "clip.wav")
    assert clip["sr"] == 44_100
    assert clip["channels"] == 1
    assert clip["bitdepth"] == 16
    assert clip["is_clipped"] is True
    assert clip["clipped_samples"] > 0
    assert 2.9 < clip["duration_s"] < 3.1


def test_clean_clip_not_flagged(mini_dataset):
    clip = health.measure_clip(mini_dataset / "a.wav")
    assert clip["is_clipped"] is False
    # ~110 Hz tone → guarded median lands near the fundamental.
    assert clip["f0_median_hz"] is not None
    assert 95 < clip["f0_median_hz"] < 125
    assert clip["voiced_frac"] > 0.5


def test_noise_floor_and_snr(mini_dataset):
    clean = health.measure_clip(mini_dataset / "a.wav")
    noisy = health.measure_clip(mini_dataset / "noisy.wav")
    # The noisy clip has a much higher (less negative) noise floor than the dry clean tone.
    assert noisy["noise_floor_dbfs"] > clean["noise_floor_dbfs"]
    # And a worse SNR proxy.
    assert noisy["snr_db"] < clean["snr_db"]


def test_dryness_dry_clip_is_negative(mini_dataset):
    clean = health.measure_clip(mini_dataset / "a.wav")
    # Hard offset into silence → tail collapses far below the speech energy.
    assert clean["dryness_db"] is not None
    assert clean["dryness_db"] < -12


def test_stereo_48k_conformance(stereo_48k):
    clip = health.measure_clip(stereo_48k / "s.wav")
    assert clip["channels"] == 2
    assert clip["sr"] == 48_000
    assert clip["bitdepth"] == 24


def test_aggregate_minutes_and_format(mini_dataset):
    report = health.measure_dataset(mini_dataset)
    m = report["metrics"]
    assert m["clip_count"] == 4
    # The clipped clip is excluded from usable minutes.
    assert m["usable_clip_count"] == 3
    assert m["total_minutes"] > m["usable_minutes"]
    assert report["format"]["sr_counts"] == {"44100": 4}
    assert report["format"]["channel_counts"] == {"1": 4}
    assert 0.0 <= m["frac_clipped"] <= 1.0
    assert m["frac_clipped"] > 0  # the clip.wav

    # Pitch coverage spans the 110-180 Hz spread across clips.
    assert report["pitch"]["have_data"] is True
    assert report["pitch"]["span_semitones"] >= 1


def test_report_is_json_safe(mini_dataset):
    report = health.measure_dataset(mini_dataset)
    report.pop("clips", None)
    _no_numpy_scalars(report)
    json.dumps(report)  # must not raise


def test_no_transcripts_means_unknown_phone_coverage(mini_dataset):
    report = health.measure_dataset(mini_dataset)
    assert report["phones"]["have_transcripts"] is False
    assert report["metrics"]["phone_coverage_weighted_pct"] is None
