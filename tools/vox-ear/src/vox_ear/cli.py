"""`vox ear` CLI — voice-native descriptors as an smpl-protocol `feature` frame.

    smpl read v.wav | vox ear describe             → feature frame (voice.* keys), passthrough
    vox ear describe --in v.wav                    → standalone (no upstream `smpl read`)

Reads NDJSON frames on stdin, resolves the last-wins `audio` frame from the CAS, and appends
one `feature` frame (role ``ear:<role>``) carrying the registered ``voice.*`` descriptor keys.
Heavy dep is praat-parselmouth → isolated venv, reached via `vox ear`.
"""

from __future__ import annotations

import argparse
import io
import sys

from . import descriptors


def _pm_version() -> str:
    try:
        import parselmouth
        return getattr(parselmouth, "VERSION", "unknown")
    except Exception:  # noqa: BLE001  # pragma: no cover
        return "absent"


def _read_stdin_frames():
    if sys.stdin.isatty():
        return []
    data = sys.stdin.buffer.read()
    if not data.strip():
        return []
    from smplstream import ndjson
    return list(ndjson.read_frames(io.BytesIO(data)))


def _resolve_audio(frames, in_path):
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
                           op_version="ear-in@1", params={"path": in_path}, fmt=meta.get("fmt"))
        return data.mean(axis=1), int(sr), fr
    return None, None, None


def _with_source(frames, src):
    """Ensure a ``--in`` ingest's source frame is emitted (lineage closure + downstream resolve)."""
    if src and src.get("id") and not any(f.get("id") == src["id"] for f in frames):
        return list(frames) + [src]
    return list(frames)


def main(argv=None) -> int:
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, ValueError, AttributeError):
        pass

    p = argparse.ArgumentParser(prog="vox ear", description="voice-native descriptors")
    sub = p.add_subparsers(dest="cmd")
    d = sub.add_parser("describe", help="emit voice.* descriptor feature frame")
    d.add_argument("--in", dest="in_path", help="WAV file (standalone; else read frames on stdin)")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.cmd is None:
        args.cmd, args.in_path = "describe", getattr(args, "in_path", None)

    frames = _read_stdin_frames()
    x, sr, src = _resolve_audio(frames, getattr(args, "in_path", None))

    from smplstream import error_frame, ndjson
    from smplstream import frames as F
    if x is None:
        sys.stderr.write("vox ear: no resolvable audio frame on stdin (and no --in FILE)\n")
        ndjson.write_frames(list(frames) + [error_frame("not_found", "no audio frame", op="ear")])
        sys.stdout.buffer.flush()
        return 1
    frames = _with_source(frames, src)  # close lineage for a --in ingest

    try:
        data = descriptors.describe(x, sr)
    except Exception as exc:  # noqa: BLE001 — one frame, one failure
        sys.stderr.write(f"vox ear: {exc}\n")
        ndjson.write_frames(list(frames) + [error_frame("op_failed", str(exc),
                            of=(src or {}).get("id"), op="ear")])
        sys.stdout.buffer.flush()
        return 1

    role = f"ear:{(src.get('role') if src else None) or 'voice'}"
    feat = F.feature_frame(data, role=role, of=(src or {}).get("id"),
                           lineage=[src["id"]] if src and src.get("id") else None,
                           op="ear-describe",
                           op_version=f"{descriptors.DESCRIBE_OP_VERSION}:parselmouth-{_pm_version()}",
                           params={"sr_hz": sr})
    ndjson.write_frames(list(frames) + [feat])
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
