"""Rubric-engine tests — operator semantics, warn zones, na handling, score math, critical fail.

These build report dicts directly (no audio) so the math is exercised in isolation."""

from __future__ import annotations

from vox_dataset import rubric


def _report(**metrics):
    """A minimal report skeleton with the given scalar metrics."""
    return {
        "metrics": metrics,
        "lists": {},
        "format": {},
        "phones": {"coverage": {"missing": []}},
        "pitch": {},
    }


def test_ge_pass_warn_fail():
    checks = [{"id": "m", "metric": "x", "op": "ge", "target": 30, "warn": 20, "weight": 1}]
    prof = {"name": "t", "checks": checks}
    assert rubric.score(_report(x=35), prof)["checks"][0]["status"] == "pass"
    assert rubric.score(_report(x=25), prof)["checks"][0]["status"] == "warn"
    assert rubric.score(_report(x=10), prof)["checks"][0]["status"] == "fail"


def test_le_operator():
    checks = [{"id": "clip", "metric": "frac_clipped", "op": "le", "target": 0.0, "warn": 0.05,
               "weight": 1}]
    prof = {"name": "t", "checks": checks}
    assert rubric.score(_report(frac_clipped=0.0), prof)["checks"][0]["status"] == "pass"
    assert rubric.score(_report(frac_clipped=0.02), prof)["checks"][0]["status"] == "warn"
    assert rubric.score(_report(frac_clipped=0.5), prof)["checks"][0]["status"] == "fail"


def test_na_when_metric_none_excluded_from_score():
    checks = [
        {"id": "ok", "metric": "x", "op": "ge", "target": 10, "weight": 1},
        {"id": "cov", "metric": "phone_coverage_weighted_pct", "op": "ge", "target": 95, "weight": 3},
    ]
    prof = {"name": "t", "checks": checks}
    r = rubric.score(_report(x=20, phone_coverage_weighted_pct=None), prof)
    # The na check is dropped from the denominator: a single passing check -> 100.
    assert r["counts"]["na"] == 1
    assert r["score"] == 100.0


def test_score_math_weighted():
    checks = [
        {"id": "a", "metric": "a", "op": "ge", "target": 1, "weight": 3},   # pass -> 3*1.0
        {"id": "b", "metric": "b", "op": "ge", "target": 100, "warn": 50, "weight": 1},  # warn 25 -> 0.5
        {"id": "c", "metric": "c", "op": "ge", "target": 100, "weight": 1},  # fail -> 0
    ]
    prof = {"name": "t", "checks": checks}
    r = rubric.score(_report(a=5, b=60, c=0), prof)
    # (3*1.0 + 1*0.5 + 1*0.0) / (3+1+1) = 3.5/5 = 70.
    assert r["score"] == 70.0


def test_critical_fail_sets_flag_and_grade():
    checks = [{"id": "min", "metric": "usable_minutes", "op": "ge", "target": 30,
               "weight": 1, "severity": "critical"}]
    prof = {"name": "t", "checks": checks}
    r = rubric.score(_report(usable_minutes=5), prof)
    assert r["critical_fail"] is True
    assert r["grade"] == "BLOCKED"


def test_dist_frac():
    checks = [{"id": "mono", "metric": "channel_counts", "op": "dist_frac", "key": "1",
               "min": 1.0, "warn": 0.9, "weight": 1}]
    prof = {"name": "t", "checks": checks}
    rep = _report()
    rep["format"]["channel_counts"] = {"1": 10}
    assert rubric.score(rep, prof)["checks"][0]["status"] == "pass"
    rep["format"]["channel_counts"] = {"1": 9, "2": 1}
    assert rubric.score(rep, prof)["checks"][0]["status"] == "warn"
    rep["format"]["channel_counts"] = {"1": 5, "2": 5}
    assert rubric.score(rep, prof)["checks"][0]["status"] == "fail"


def test_frac_in_range():
    checks = [{"id": "seg", "metric": "durations", "op": "frac_in_range", "target": [5, 15],
               "min": 0.8, "warn": 0.5, "weight": 1}]
    prof = {"name": "t", "checks": checks}
    rep = _report()
    rep["lists"]["durations"] = [6, 7, 8, 9, 10]     # all in range
    assert rubric.score(rep, prof)["checks"][0]["status"] == "pass"
    rep["lists"]["durations"] = [6, 7, 20, 30]       # half in range -> warn (>=0.5)
    assert rubric.score(rep, prof)["checks"][0]["status"] == "warn"
    rep["lists"]["durations"] = [20, 30, 40, 6]      # only 1/4 -> fail
    assert rubric.score(rep, prof)["checks"][0]["status"] == "fail"


def test_remediation_fills_missing_phones():
    checks = [{"id": "cov", "metric": "phone_coverage_weighted_pct", "op": "ge", "target": 95,
               "weight": 1, "remediation": "missing: {missing}"}]
    prof = {"name": "t", "checks": checks}
    rep = _report(phone_coverage_weighted_pct=80)
    rep["phones"]["coverage"] = {"missing": ["OY", "ZH"]}
    c = rubric.score(rep, prof)["checks"][0]
    assert "OY" in c["remediation"] and "ZH" in c["remediation"]


def test_all_shipped_profiles_load_and_score():
    # Every profile in the package must parse and score a real (empty-ish) report without error.
    rep = _report(usable_minutes=45, frac_clipped=0.0, mean_snr_db=45,
                  median_noise_floor_dbfs=-60, median_dryness_db=-35,
                  pitch_span_semitones=15, phone_coverage_weighted_pct=97, clip_count=100)
    rep["format"]["channel_counts"] = {"1": 100}
    rep["format"]["sr_counts"] = {"44100": 100}
    rep["lists"]["durations"] = [8.0] * 100
    rep["lists"]["syllable_rate_hz"] = [4.0] * 100
    for name in rubric.list_profiles():
        r = rubric.score(rep, name)
        assert 0.0 <= r["score"] <= 100.0
        assert r["profile"] == name


def test_profiles_present():
    names = rubric.list_profiles()
    assert "diffsinger-acoustic" in names
    assert "vocoder-ft" in names
    assert "styletts2-ft" in names
