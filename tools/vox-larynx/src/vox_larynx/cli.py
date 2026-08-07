"""`vox larynx` CLI — an smpl-protocol WORLD-vocoder bridge (analyze / render / harmonize).

    smpl read v.wav | vox larynx analyze                  → a `feature` frame (F0 skeleton stats)
    smpl read v.wav | vox larynx render --semitones 5     → retuned `audio` frame (formants kept)
    smpl read v.wav | vox larynx render --to-note A3       → monotone at A3 (pitch imposition)
    smpl read v.wav | vox larynx harmonize --chord 0,3,7 --drone    → stacked-choir `audio` frame

Frame citizen: reads NDJSON frames on stdin, resolves the last-wins `audio` frame from the CAS,
passes every input frame through unchanged, then appends its derived frame(s). For standalone
use (no upstream `smpl read`), ``--in FILE`` loads a WAV directly. The heavy dep is **pyworld**
(a compiled wheel), so this tool lives in its own isolated venv — reached via `vox larynx`.
"""

from __future__ import annotations

import argparse
import io
import sys

from . import world

OP_PREFIX = "larynx"


def _pyworld_version() -> str:
    try:
        import pyworld
        return getattr(pyworld, "__version__", "unknown")
    except Exception:  # noqa: BLE001  # pragma: no cover - import guard
        return "absent"


def _op_version(base: str) -> str:
    return f"{base}:pyworld-{_pyworld_version()}"


def _read_stdin_frames():
    if sys.stdin.isatty():
        return []
    data = sys.stdin.buffer.read()
    if not data.strip():
        return []
    from smplstream import ndjson
    return list(ndjson.read_frames(io.BytesIO(data)))


def _resolve_audio(frames, in_path):
    """Return (samples float64 mono, sr, src_frame). Prefer the last-wins upstream audio frame;
    fall back to ``--in FILE`` (CAS'd as a fresh source frame so lineage still closes)."""
    import soundfile as sf
    from smplstream import cas
    from smplstream.select import select

    audio = select(frames, kind="audio", predicate=lambda f: bool(f.get("hash")), mode="last")
    if audio:
        fr = audio[0]
        data, sr = sf.read(str(cas.get_path(fr["hash"])), dtype="float64", always_2d=True)
        return data.mean(axis=1), int(sr), fr
    if in_path:
        data, sr = sf.read(in_path, dtype="float64", always_2d=True)
        buf = io.BytesIO()
        sf.write(buf, data.astype("float32"), sr, format="WAV", subtype="FLOAT")
        h = cas.put_audio_bytes(buf.getvalue())
        meta = cas.read_meta(h) or {}
        from smplstream import frames as F
        fr = F.audio_frame(h, sr=meta.get("sr", sr), ch=meta.get("ch", data.shape[1]),
                           dur=meta.get("dur", 0.0), role="source", op="from-file",
                           op_version="larynx-in@1", params={"path": in_path}, fmt=meta.get("fmt"))
        return data.mean(axis=1), int(sr), fr
    return None, None, None


def _with_source(frames, src):
    """Ensure the source frame is in the passthrough — so a ``--in FILE`` ingest doesn't leave
    the derived frame's ``of``/``lineage`` dangling (and downstream can resolve the audio)."""
    if src and src.get("id") and not any(f.get("id") == src["id"] for f in frames):
        return list(frames) + [src]
    return list(frames)


def _wet_role(src_frame) -> str:
    role = (src_frame.get("role") if src_frame else None) or "voice"
    for suf in (".wet", ".dry"):
        role = role.removesuffix(suf)
    return f"{role}.wet"


def _emit_audio(samples, sr, *, src_frame, op, op_version, params, passthrough):
    import numpy as np
    import soundfile as sf
    from smplstream import cas, ndjson
    from smplstream import frames as F

    arr = np.ascontiguousarray(np.asarray(samples, dtype="float32"))
    if arr.ndim == 1:
        arr = arr[:, None]
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    frame = F.audio_frame(
        h, sr=meta.get("sr", sr), ch=meta.get("ch", arr.shape[1]),
        dur=meta.get("dur", arr.shape[0] / sr if sr else 0.0),
        role=_wet_role(src_frame), of=(src_frame or {}).get("id"),
        lineage=[src_frame["id"]] if src_frame and src_frame.get("id") else None,
        op=op, op_version=op_version, params=params, fmt=meta.get("fmt"))
    out = list(passthrough) + [frame]
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()


def _emit_feature(data, *, src_frame, op, op_version, params, passthrough):
    from smplstream import frames as F
    from smplstream import ndjson

    role = f"larynx:{(src_frame.get('role') if src_frame else None) or 'voice'}"
    frame = F.feature_frame(data, role=role, of=(src_frame or {}).get("id"),
                            lineage=[src_frame["id"]] if src_frame and src_frame.get("id") else None,
                            op=op, op_version=op_version, params=params)
    out = list(passthrough) + [frame]
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()


def _no_audio(passthrough) -> int:
    from smplstream import error_frame, ndjson
    sys.stderr.write("vox larynx: no resolvable audio frame on stdin (and no --in FILE)\n")
    out = list(passthrough) + [error_frame("not_found", "no audio frame to process", op=OP_PREFIX)]
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    return 1


def _op_failed(exc, src_frame, passthrough) -> int:
    from smplstream import error_frame, ndjson
    sys.stderr.write(f"vox larynx: {exc}\n")
    out = list(passthrough) + [error_frame("op_failed", str(exc),
                                            of=(src_frame or {}).get("id"), op=OP_PREFIX)]
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    return 1


def _build_parser():
    p = argparse.ArgumentParser(prog="vox larynx", description="WORLD-vocoder voice tier")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="emit an F0/skeleton feature frame")
    a.add_argument("--in", dest="in_path", help="WAV file (standalone; else read frames on stdin)")

    r = sub.add_parser("render", help="retune / re-voice (replace F0, keep formants)")
    r.add_argument("--in", dest="in_path")
    r.add_argument("--semitones", type=float, default=0.0, help="transpose by N semitones")
    r.add_argument("--to-note", help="force every voiced frame to this note (e.g. A3)")
    r.add_argument("--to-hz", type=float, help="force every voiced frame to this Hz")
    r.add_argument("--flatten", action="store_true", help="monotone at the median F0")
    r.add_argument("--formant-shift", type=float, default=1.0, help="formant warp ratio (>1 up)")
    r.add_argument("--stretch", type=float, default=1.0, help="time-scale ratio (>1 slower)")

    h = sub.add_parser("harmonize", help="chord-locked choir stack, formants preserved")
    h.add_argument("--in", dest="in_path")
    h.add_argument("--chord", default="0,3,7", help="comma semitone offsets (default minor triad)")
    h.add_argument("--mode", choices=("follow", "flat"), default="follow",
                   help="follow=natural choir; flat=fixed-pitch sheen")
    h.add_argument("--no-lead", action="store_true", help="omit the dry lead from the mix")
    h.add_argument("--drone", action="store_true", help="add a sustained tonic drone")
    h.add_argument("--drone-octave", type=int, default=-1)
    h.add_argument("--spread", type=float, default=18.0, help="vibrato extent (cents)")
    h.add_argument("--jitter", type=float, default=0.004, help="per-voice F0 jitter fraction")
    h.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> int:
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, ValueError, AttributeError):
        pass

    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    frames = _read_stdin_frames()
    x, sr, src = _resolve_audio(frames, getattr(args, "in_path", None))
    if x is None:
        return _no_audio(frames)
    frames = _with_source(frames, src)  # close lineage for a --in ingest

    try:
        if args.cmd == "analyze":
            f0, _sp, _ap, _t = world.analyze(x, sr)
            data = world.f0_stats(f0)
            _emit_feature(data, src_frame=src, op="larynx-analyze",
                          op_version=_op_version(world.ANALYZE_OP_VERSION),
                          params={"frame_period_ms": world.FRAME_PERIOD, "sr_hz": sr},
                          passthrough=frames)
        elif args.cmd == "render":
            to_hz = args.to_hz
            if args.to_note:
                to_hz = world.note_to_hz(args.to_note)
            y, params = world.render(x, sr, semitones=args.semitones, to_hz=to_hz,
                                     flatten=args.flatten, formant_ratio=args.formant_shift,
                                     time_ratio=args.stretch)
            _emit_audio(y, sr, src_frame=src, op="larynx-render",
                        op_version=_op_version(world.RENDER_OP_VERSION), params=params,
                        passthrough=frames)
        elif args.cmd == "harmonize":
            chord = tuple(int(s) for s in str(args.chord).split(",") if s.strip() != "")
            y, params = world.harmonize(x, sr, chord=chord, mode=args.mode,
                                        include_lead=not args.no_lead, drone=args.drone,
                                        drone_octave=args.drone_octave, jitter=args.jitter,
                                        vib_cents=args.spread, seed=args.seed)
            _emit_audio(y, sr, src_frame=src, op="larynx-harmonize",
                        op_version=_op_version(world.HARMONIZE_OP_VERSION), params=params,
                        passthrough=frames)
    except Exception as exc:  # noqa: BLE001 — one frame, one failure
        return _op_failed(exc, src, frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
