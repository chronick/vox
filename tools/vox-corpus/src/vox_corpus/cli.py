"""`vox corpus` CLI — gate clips into a voice-conversion matching set.

    vox corpus gate clip1.wav clip2.wav ...                  → per-file JSON verdict on stdout
    vox corpus gate raw/*.wav --admit corpus/core --quarantine corpus/_quarantine
        → peak-normalized survivors written to --admit; failures copied to --quarantine

Exit code is non-zero if ANY clip failed the gate (so a batch ingest surfaces poison), while every
verdict is still printed. The gate catches the steady-state/near-silent VAD-death class that turned
metal-buzz.wav into a downstream reshape crash — before it can poison the set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import gate


def _emit(rec):
    print(json.dumps(rec))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vox corpus", description="voice-corpus ingest gate")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="peak-normalize + VAD-survivability admission check")
    g.add_argument("files", nargs="+", help="WAV clips to gate")
    g.add_argument("--admit", help="dir to write peak-normalized survivors into")
    g.add_argument("--quarantine", help="dir to copy rejected clips into")
    g.add_argument("--target-peak", type=float, default=gate.TARGET_PEAK)
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    import soundfile as sf

    any_fail = False
    admit_dir = Path(args.admit) if args.admit else None
    quar_dir = Path(args.quarantine) if args.quarantine else None
    if admit_dir:
        admit_dir.mkdir(parents=True, exist_ok=True)
    if quar_dir:
        quar_dir.mkdir(parents=True, exist_ok=True)

    for f in args.files:
        path = Path(f)
        try:
            verdict, normalized, sr = gate.gate_file(str(path))
        except Exception as exc:  # noqa: BLE001 — a bad file is a rejection, not a crash
            any_fail = True
            _emit({"file": str(path), "survives": False, "reasons": [f"read/decode failed: {exc}"]})
            continue

        rec = {"file": str(path), **verdict}
        _emit(rec)
        if verdict["survives"]:
            if admit_dir:
                dest = admit_dir / path.name
                sf.write(str(dest), normalized, sr, subtype="FLOAT")
                rec["admitted_to"] = str(dest)
        else:
            any_fail = True
            if quar_dir:
                shutil.copyfile(str(path), quar_dir / path.name)

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
