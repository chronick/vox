"""`vox take` CLI — render + self-verify a take-card.

    vox take run --card machinery.yaml --out take.wav        → audio + a verification report frame
    vox take run --card machinery.yaml --json                → print the report as JSON

Emits an `audio` frame (the render, CAS'd) and a `feature` frame (role ``take:<name>``) carrying
the measured axis coordinate and — when the card has a `target:` — the programmed-vs-measured
per-axis error. That number is the self-verifying take: how far the render landed from the
coordinate the card asked for.
"""

from __future__ import annotations

import argparse
import io
import json
import sys

from . import orchestrate


def main(argv=None) -> int:
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, ValueError, AttributeError):
        pass

    p = argparse.ArgumentParser(prog="vox take", description="render + self-verify a take-card")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="render a take-card and verify it against its target")
    r.add_argument("--card", required=True, help="take-card YAML")
    r.add_argument("--out", help="also write the rendered WAV here")
    r.add_argument("--json", action="store_true", help="print the report as JSON (no frames)")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    import yaml
    with open(args.card) as fh:
        card = yaml.safe_load(fh) or {}

    try:
        report = orchestrate.run_take(card)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"vox take: {exc}\n")
        from smplstream import error_frame, ndjson
        ndjson.write_frames([error_frame("op_failed", str(exc), op="take")])
        sys.stdout.buffer.flush()
        return 1

    import soundfile as sf
    y, sr = report.pop("audio"), report.pop("sr")
    if args.out:
        sf.write(args.out, y, sr, subtype="FLOAT")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    from smplstream import cas, ndjson
    from smplstream import frames as F
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    af = F.audio_frame(h, sr=meta.get("sr", sr), ch=meta.get("ch", 1),
                       dur=meta.get("dur", len(y) / sr), role=f"take:{report['name']}",
                       op="take-render", op_version="take-render@1", params=report["render"],
                       fmt=meta.get("fmt"))
    feat = F.feature_frame(report, role=f"take:{report['name']}", of=af["id"], lineage=[af["id"]],
                           op="take-verify", op_version="take-verify@1", params={})
    ndjson.write_frames([af, feat])
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
