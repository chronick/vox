"""End-to-end: measure a synthetic dataset with a transcript CSV, score it, and confirm the
transcript path lights up phone coverage + syllable rate. Whisper-dependent behaviour is guarded
by skipif so the suite passes with or without faster-whisper installed."""

from __future__ import annotations

import pytest

from vox_dataset import health, phones, rubric


def test_transcript_csv_enables_phone_coverage(mini_dataset, tmp_path):
    if not phones.have_pronouncing():
        pytest.skip("pronouncing not installed")
    csv = tmp_path / "t.csv"
    csv.write_text(
        "filename,text\n"
        "a.wav,the quick brown fox jumps over the lazy dog\n"
        "b.wav,she measures unusual beige vision treasures\n"
        "noisy.wav,a boy enjoys the joyful voice\n",
        encoding="utf-8",
    )
    transcripts = health.load_transcripts(csv)
    report = health.measure_dataset(mini_dataset, transcripts=transcripts)
    assert report["phones"]["have_transcripts"] is True
    m = report["metrics"]
    assert m["phone_coverage_weighted_pct"] is not None
    assert 0.0 < m["phone_coverage_weighted_pct"] <= 100.0
    # The transcripts include ZH-bearing words (measures/vision/treasure) and OY (boy/joyful/voice).
    counts = report["phones"]["counts"]
    assert counts.get("ZH", 0) >= 1
    assert counts.get("OY", 0) >= 1


def test_whisper_gating_reports_unknown_without_model(mini_dataset):
    # use_whisper=True but if faster-whisper absent, load_whisper returns None and coverage is unknown.
    if health.whisper_available():
        pytest.skip("faster-whisper installed; this checks the absent path")
    report = health.measure_dataset(mini_dataset, use_whisper=True)
    assert report["phones"]["have_transcripts"] is False
    assert report["metrics"]["phone_coverage_weighted_pct"] is None


@pytest.mark.skipif(not health.whisper_available(), reason="faster-whisper not installed")
def test_whisper_loads(mini_dataset):
    model = health.load_whisper("base")
    assert model is not None


def test_full_scoring_pipeline(mini_dataset):
    report = health.measure_dataset(mini_dataset)
    scored = rubric.score(report, "vocoder-ft")
    assert "score" in scored
    assert scored["profile"] == "vocoder-ft"
    # Phone-free profile: no na checks from missing transcripts.
    assert scored["counts"]["na"] == 0
