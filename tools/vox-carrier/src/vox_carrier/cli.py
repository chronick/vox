"""`vox carrier` CLI — pour spat words into a deep harsh body.

    vox carrier one --pattern "X.x.X.x." --syllables "da,ka,ta,ma" --body growl-55 --out take.wav
    vox carrier verse --lines "Kick the pattern back to the top|Cut the deck and count to ten" \\
        --body growl-55 --bpm 142 --out verse.wav

Needs `say` + `ffmpeg` (the FLOW render deps) and, for SC bodies, SuperCollider.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import carrier


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vox carrier", description="deep-body carrier voice")
    sub = p.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("one", help="one authored FLOW pattern through a body")
    o.add_argument("--pattern", required=True)
    o.add_argument("--syllables", required=True, help="comma-separated")
    o.add_argument("--body", required=True, help="a bodies.yaml entry (e.g. growl-55)")
    o.add_argument("--bpm", type=float, default=140.0)
    o.add_argument("--ess-mix", type=float, default=0.30)
    o.add_argument("--out", required=True)

    v = sub.add_parser("verse", help="lyric lines spat per-bar through a body")
    v.add_argument("--lines", required=True, help="'|'-separated lyric lines")
    v.add_argument("--body", required=True)
    v.add_argument("--bpm", type=float, default=140.0)
    v.add_argument("--ess-mix", type=float, default=0.30)
    v.add_argument("--dry-db", type=float, default=-14.0,
                   help="dry-diction layer level rel. body RMS (use 99 to disable)")
    v.add_argument("--out", required=True)

    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    try:
        import soundfile as sf
        if args.cmd == "one":
            r = carrier.render_carrier_one(args.pattern,
                                           [s.strip() for s in args.syllables.split(",") if s.strip()],
                                           args.bpm, args.body, ess_mix=args.ess_mix)
        else:
            dry_db = None if args.dry_db >= 90 else args.dry_db
            r = carrier.render_carrier_verse(args.lines.split("|"), args.bpm, args.body,
                                             ess_mix=args.ess_mix, dry_db=dry_db)
        sf.write(args.out, r["final"], r["sr"], subtype="FLOAT")
        print(json.dumps({"out": args.out, "sr": r["sr"],
                          "dur_s": round(len(r["final"]) / r["sr"], 3),
                          "params": r["params"]}, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"vox carrier: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
