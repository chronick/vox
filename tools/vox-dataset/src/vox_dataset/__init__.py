"""vox-dataset — the dataset-health doctor.

Measures a directory of audio clips for voicebank fitness and scores it against a
per-target-model rubric. Public surface:

    from vox_dataset import health, rubric
    report = health.measure_dataset("/path/to/clips")      # per-clip + aggregate metrics
    scored = rubric.score(report, "diffsinger-acoustic")   # 0-100 + per-check pass/warn/fail

faster-whisper is an optional import: without transcripts, phone coverage is reported as
``unknown`` and rubric checks that need it become ``na`` (excluded from the score).
"""

from __future__ import annotations

__all__ = ["health", "rubric"]

__version__ = "0.1.0"
