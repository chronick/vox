"""`vox cast` CLI — voice conversion through a trained RVC model, as an smpl stage.

    vox cast setup                                   → build the engine venv (one-time)
    smpl read in.wav | vox cast convert --model DIR --trust-model | smpl write out.wav
    vox cast convert --in take.wav --model growl --trust-model | smpl write out.wav
    vox cast info --model growl                      → describe a cast from disk
    vox cast import --model ~/Downloads/growl        → copy a local cast into the library
    vox cast list                                    → list library casts as JSON

Frame citizen: reads NDJSON frames on stdin, resolves the last-wins `audio`
frame from the CAS, passes every input frame through unchanged, then appends
the converted frame. The ML stack never loads into this process — conversion
shells out to the engine venv (see engine.py); if that venv is missing, the
command degrades to a clear install hint instead of a traceback.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys

from . import engine, model

OP_PREFIX = "cast"
_DEVICE = re.compile(r"^(?:cpu(?::[0-9]+)?|cuda:[0-9]+|mps)$")


def _bounded_float(option: str, minimum: float, maximum: float):
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{option} must be a number") from exc
        if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{option} must be a finite number from {minimum:g} to {maximum:g}"
            )
        return parsed

    return parse


def _device(value: str) -> str:
    if not _DEVICE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "--device must be cpu, cpu:N, cuda:N, or mps (N is a non-negative integer)"
        )
    return value


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
                           op_version="cast-in@1", params={"path": in_path}, fmt=meta.get("fmt"))
        return data.mean(axis=1), int(sr), fr
    return None, None, None


def _with_source(frames, src):
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


def _fail(code, message, *, src_frame=None, passthrough=()) -> int:
    from smplstream import error_frame, ndjson
    sys.stderr.write(f"vox cast: {message}\n")
    out = list(passthrough) + [error_frame(code, message,
                                            of=(src_frame or {}).get("id"), op=OP_PREFIX)]
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    return 1


def _convert(args) -> int:
    import soundfile as sf

    frames = _read_stdin_frames()
    x, sr, src = _resolve_audio(frames, args.in_path)
    if x is None:
        return _fail("not_found", "no resolvable audio frame on stdin (and no --in FILE)",
                     passthrough=frames)
    frames = _with_source(frames, src)

    try:
        cast = model.resolve_cast(args.model, index=args.index)
    except model.CastNotFound as exc:
        return _fail("not_found", str(exc), src_frame=src, passthrough=frames)

    if not engine.is_installed():
        return _fail(
            "op_failed",
            f"engine not installed (looked in {engine.engine_dir()}) — {engine.INSTALL_HINT}",
            src_frame=src, passthrough=frames)

    if not args.trust_model:
        return _fail(
            "op_failed",
            "refusing to load untrusted RVC weights: .pth models are Python pickle files "
            "that may execute code when loaded, and the engine venv is not a security "
            "boundary; inspect the model and rerun with --trust-model only if you trust "
            "its source",
            src_frame=src, passthrough=frames)

    params = {
        "model": cast["name"], "pth": cast["pth"].name,
        "pitch": args.pitch, "f0_method": args.f0_method,
        "index_rate": args.index_rate, "protect": args.protect,
        "rms_mix": args.rms_mix, "arch": args.arch, "device": args.device,
    }

    import tempfile
    with tempfile.TemporaryDirectory(prefix="vox-cast-") as td:
        from pathlib import Path
        in_wav, out_wav = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(in_wav), x.astype("float32"), sr, subtype="FLOAT")
        rc, err = engine.run_convert(in_wav, out_wav, pth=cast["pth"], index=cast["index"],
                                     params=params, cwd=Path(td))
        if rc != 0 or not out_wav.is_file():
            tail = "\n".join(err.strip().splitlines()[-6:])
            return _fail("op_failed", f"engine conversion failed (exit {rc})\n{tail}",
                         src_frame=src, passthrough=frames)
        y, out_sr = sf.read(str(out_wav), dtype="float64", always_2d=True)

    _emit_audio(y.mean(axis=1), int(out_sr), src_frame=src, op="cast-convert",
                op_version=f"cast-convert@1:rvc-python-{engine.rvc_version()}",
                params=params, passthrough=frames)
    return 0


def _info(args) -> int:
    try:
        cast = model.resolve_cast(args.model, index=args.index)
    except model.CastNotFound as exc:
        sys.stderr.write(f"vox cast: {exc}\n")
        return 1
    print(json.dumps(model.cast_info(cast, checksums=True), indent=2))
    return 0


def _import(args) -> int:
    try:
        cast = model.import_cast(args.model, index=args.index, name=args.name)
    except (model.CastImportError, OSError) as exc:
        sys.stderr.write(f"vox cast: {exc}\n")
        return 1
    print(json.dumps(model.cast_info(cast, checksums=True), indent=2))
    return 0


def _list(_args) -> int:
    print(json.dumps(model.list_casts(), indent=2))
    return 0


def _setup(args) -> int:
    if args.status:
        print(json.dumps(engine.status(), indent=2))
        return 0
    return engine.setup(fresh=args.fresh)


def _build_parser():
    p = argparse.ArgumentParser(prog="vox cast",
                                description="voice conversion through a trained RVC model")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="build the isolated engine venv (one-time, ~3 GB)")
    s.add_argument("--status", action="store_true", help="report engine state, change nothing")
    s.add_argument("--fresh", action="store_true", help="rebuild the venv from scratch")

    c = sub.add_parser("convert", help="convert audio through a cast (smpl pipe stage)")
    c.add_argument("--in", dest="in_path", help="WAV file (standalone; else read frames on stdin)")
    c.add_argument("--model", required=True,
                   help="cast name (under ~/.vox/casts), model dir, or .pth path")
    c.add_argument("--index", help="explicit .index path (default: the dir's single .index)")
    c.add_argument("--pitch", type=int, default=0, help="transpose in semitones (RVC f0up_key)")
    c.add_argument("--f0-method", choices=("rmvpe", "harvest", "crepe"), default="rmvpe",
                   help="f0 estimator (rmvpe|harvest|crepe)")
    c.add_argument("--index-rate", type=_bounded_float("--index-rate", 0.0, 1.0), default=0.5,
                   help="retrieval blend 0–1 (higher = more of the cast's timbre bank)")
    c.add_argument("--protect", type=_bounded_float("--protect", 0.0, 0.5), default=0.33,
                   help="consonant/breath protection 0–0.5 (lower = more conversion)")
    c.add_argument("--rms-mix", type=_bounded_float("--rms-mix", 0.0, 1.0), default=1.0,
                   help="loudness envelope blend 0–1 (1 = keep the source dynamics)")
    c.add_argument("--arch", choices=("v1", "v2"), default="v2", help="RVC model generation")
    c.add_argument("--device", type=_device, default="cpu:0",
                   help="cpu:0 (default, deterministic) | cuda:0 | mps (opt-in, unstable)")
    c.add_argument(
        "--trust-model", action="store_true",
        help="load this trusted .pth pickle despite its ability to execute code",
    )

    i = sub.add_parser("info", help="describe a cast from its files (no ML stack needed)")
    i.add_argument("--model", required=True)
    i.add_argument("--index")

    a = sub.add_parser("import", help="copy an authorized local model into the cast library")
    a.add_argument("--model", required=True, help="local .pth file or model directory")
    a.add_argument("--index", help="explicit local .index file")
    a.add_argument("--name", help="library name (default: directory or .pth stem)")

    sub.add_parser("list", help="list imported casts as JSON")
    return p


def main(argv=None) -> int:
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, ValueError, AttributeError):
        pass

    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.cmd == "setup":
        return _setup(args)
    if args.cmd == "info":
        return _info(args)
    if args.cmd == "import":
        return _import(args)
    if args.cmd == "list":
        return _list(args)
    return _convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
