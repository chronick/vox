"""`vox bodies` CLI — list, render, and fingerprint the carrier-voice palette.

    vox bodies list                                  → the palette as JSON
    vox bodies render growl-55 --out growl.wav       → render one body to a file
    vox bodies fingerprint [--no-write]              → re-measure every fingerprint
"""

from __future__ import annotations

import argparse
import json
import sys

from . import registry


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vox bodies", description="carrier-voice palette")
    sub = p.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("list", help="print the validated palette as JSON")
    ls.add_argument("--palette", help="palette yaml (default: the shipped bodies.yaml)")

    r = sub.add_parser("render", help="render one body to a WAV")
    r.add_argument("name")
    r.add_argument("--out", required=True)
    r.add_argument("--dur", type=float, default=2.0)
    r.add_argument("--palette")

    f = sub.add_parser("fingerprint", help="render + measure every body, write back")
    f.add_argument("--no-write", action="store_true", help="measure only; leave the yaml alone")
    f.add_argument("--dur", type=float, default=2.0)
    f.add_argument("--palette")

    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    try:
        if args.cmd == "list":
            print(json.dumps(registry.load_bodies(args.palette), indent=2))
            return 0
        if args.cmd == "render":
            import soundfile as sf
            y, sr = registry.render_body(args.name, dur=args.dur, path=args.palette)
            sf.write(args.out, y, sr, subtype="FLOAT")
            print(json.dumps({"name": args.name, "out": args.out, "sr": sr,
                              "dur_s": round(len(y) / sr, 3)}))
            return 0
        # fingerprint
        out = registry.measure_fingerprints(path=args.palette, dur=args.dur,
                                            write=not args.no_write)
        print(json.dumps(out, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"vox bodies: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
