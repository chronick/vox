"""`vox flow` CLI — the FLOW cadence grammar (score) and spat rendering (render).

    vox flow score --pattern "X.x.X.x.xxx.X..." --bpm 140 --grid 4           → marker frame (onsets)
    vox flow render --pattern "X..x..X.x." --syllables "da,ka,ta,ma" --chain grit
        → an `audio` frame: `say` syllables placed on the grid, driven through the grit chain

`render` is a SOURCE tool (audio from a pattern, no upstream audio). Both pass any input frames
through. `say`/`ffmpeg` are system binaries — absent → a clean `unsupported` error frame.
"""

from __future__ import annotations

import argparse
import io
import sys

from . import flow, render


def _read_stdin_frames():
    if sys.stdin.isatty():
        return []
    data = sys.stdin.buffer.read()
    if not data.strip():
        return []
    from smplstream import ndjson
    return list(ndjson.read_frames(io.BytesIO(data)))


def _add_common(p):
    p.add_argument("--pattern", required=True, help="grid pattern (x/X onset, ./space rest, -/_ hold)")
    p.add_argument("--bpm", type=float, default=140.0)
    p.add_argument("--grid", type=int, default=4, help="steps per beat (4=16ths, 3=8th-triplets)")
    p.add_argument("--swing", type=float, default=0.0)
    p.add_argument("--push-ms", type=float, default=0.0, help="micro-timing: -rush / +drag")
    p.add_argument("--syllables", default="", help="comma-separated syllables assigned to onsets")


def _syllables(arg):
    return [s.strip() for s in arg.split(",") if s.strip()]


def main(argv=None) -> int:
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, ValueError, AttributeError):
        pass

    p = argparse.ArgumentParser(prog="vox flow", description="cadence grammar / spat render")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="compile a pattern to onset events (marker frame)")
    _add_common(s)
    r = sub.add_parser("render", help="render the pattern spat via say, through an fx chain")
    _add_common(r)
    r.add_argument("--voice", default="Fred", help="macOS `say` voice (default Fred — robotic)")
    r.add_argument("--pbas", type=float, default=92.0, help="say pitch base in Hz")
    r.add_argument("--chain", choices=(*flow.CHAINS, "none"), default="grit")
    r.add_argument("--sr", type=int, default=44100)
    r.add_argument("--role", default="voice")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    frames = _read_stdin_frames()
    score = flow.compile_flow(args.pattern, bpm=args.bpm, grid=args.grid, swing=args.swing,
                              push_ms=args.push_ms, syllables=_syllables(args.syllables))

    from smplstream import error_frame, ndjson
    from smplstream import frames as F

    if args.cmd == "score":
        points = [{"t": e["t"], "dur": e["dur"], "label": (e["syllable"] or f"step{e['step']}")}
                  for e in score["events"]]
        mk = F.marker_frame(points, role="flow:onset", op="flow-score", op_version="flow-score@1",
                            params={k: score[k] for k in ("bpm", "grid", "swing", "push_ms",
                                                           "n_onsets", "bar_seconds")})
        ndjson.write_frames(list(frames) + [mk])
        sys.stdout.buffer.flush()
        return 0

    # render
    if not render.say_available():
        sys.stderr.write("vox flow: `say` not on PATH (macOS speech synth required)\n")
        ndjson.write_frames(list(frames) + [error_frame("unsupported", "say not found", op="flow")])
        sys.stdout.buffer.flush()
        return 0
    try:
        y = render.render_flow(score, voice=args.voice, pbas=args.pbas, sr=args.sr)
        chain_applied = None
        if args.chain != "none":
            y = render.apply_chain(y, args.sr, args.chain)
            chain_applied = args.chain
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"vox flow: {exc}\n")
        ndjson.write_frames(list(frames) + [error_frame("op_failed", str(exc), op="flow")])
        sys.stdout.buffer.flush()
        return 1

    import soundfile as sf
    from smplstream import cas
    buf = io.BytesIO()
    sf.write(buf, y, args.sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    af = F.audio_frame(h, sr=meta.get("sr", args.sr), ch=meta.get("ch", 1),
                       dur=meta.get("dur", len(y) / args.sr), role=args.role, op="flow-render",
                       op_version="flow-render@1", params={
                           "pattern": args.pattern, "bpm": args.bpm, "grid": args.grid,
                           "swing": args.swing, "push_ms": args.push_ms, "voice": args.voice,
                           "pbas": args.pbas, "chain": chain_applied, "n_onsets": score["n_onsets"],
                           "syllables": _syllables(args.syllables)}, fmt=meta.get("fmt"))
    ndjson.write_frames(list(frames) + [af])
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
