"""`vox vector` CLI — measure a render's axis coordinate, and diff it against a target.

    smpl read v.wav | vox ear describe | vox vector measure            → vector:<role> feature frame
    smpl read v.wav | vox ear describe | vox vector diff --target '{"breathiness":0.2,"roughness":0.05}'
    vox vector diff --in v.wav --trajectory take.yaml --at 1.1         → error vs a keyframe

`measure` maps the (last-wins) audio frame — enriched by any upstream ``ear:*``/``larynx:*``
descriptor frame — onto the six axes. `diff` compares that measured coordinate to a programmed
one (inline ``--target`` JSON, or a keyframe pulled from a trajectory YAML): the mean absolute
error is the self-verifying number. Passthrough + append, like every filter.
"""

from __future__ import annotations

import argparse
import io
import json
import sys

from . import axes


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
                           op_version="vector-in@1", params={"path": in_path}, fmt=meta.get("fmt"))
        return data.mean(axis=1), int(sr), fr
    return None, None, None


def _with_source(frames, src):
    """Ensure a ``--in`` ingest's source frame is emitted (lineage closure + downstream resolve)."""
    if src and src.get("id") and not any(f.get("id") == src["id"] for f in frames):
        return list(frames) + [src]
    return list(frames)


def _upstream_voice(frames) -> dict:
    """Collect ``voice.*`` keys from the most-recent ear feature frame (if the pipe includes one)."""
    up = {}
    for f in frames:
        if f.get("kind") == "feature" and isinstance(f.get("data"), dict):
            for k, v in f["data"].items():
                if k.startswith("voice."):
                    up[k] = v
    return up


def _keyframe_axes(traj_path, at) -> dict:
    import yaml
    with open(traj_path) as fh:
        doc = yaml.safe_load(fh)
    frames = doc.get("trajectory", doc if isinstance(doc, list) else [])
    for kf in frames:
        if str(kf.get("at")) == str(at):
            return {k: kf[k] for k in axes.AXES if k in kf}
    raise ValueError(f"no keyframe at {at!r} in {traj_path}")


def main(argv=None) -> int:
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, ValueError, AttributeError):
        pass

    p = argparse.ArgumentParser(prog="vox vector", description="six-axis voice-coordinate measurement")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="map a render onto the six axes")
    m.add_argument("--in", dest="in_path")
    m.add_argument("--at", help="keyframe label to tag this measurement with (optional)")
    d = sub.add_parser("diff", help="programmed vs measured axis error")
    d.add_argument("--in", dest="in_path")
    d.add_argument("--target", help="programmed axes as inline JSON, e.g. '{\"breathiness\":0.2}'")
    d.add_argument("--trajectory", help="YAML with a `trajectory:` keyframe list")
    d.add_argument("--at", help="keyframe label to pull from --trajectory")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    frames = _read_stdin_frames()
    x, sr, src = _resolve_audio(frames, getattr(args, "in_path", None))

    from smplstream import error_frame, ndjson
    from smplstream import frames as F
    if x is None:
        sys.stderr.write("vox vector: no resolvable audio frame on stdin (and no --in FILE)\n")
        ndjson.write_frames(list(frames) + [error_frame("not_found", "no audio", op="vector")])
        sys.stdout.buffer.flush()
        return 1
    frames = _with_source(frames, src)  # close lineage for a --in ingest

    role = (src.get("role") if src else None) or "voice"
    try:
        result = axes.measure(x, sr, _upstream_voice(frames))
        if args.cmd == "measure":
            data = dict(result)
            if args.at:
                data["at"] = args.at
            out = F.feature_frame(data, role=f"vector:{role}", of=(src or {}).get("id"),
                                  lineage=[src["id"]] if src and src.get("id") else None,
                                  op="vector-measure", op_version="vector-measure@1",
                                  params={"sr_hz": sr})
        else:  # diff
            if args.trajectory and args.at:
                programmed = _keyframe_axes(args.trajectory, args.at)
            elif args.target:
                programmed = json.loads(args.target)
            else:
                sys.stderr.write("vox vector diff: need --target JSON or --trajectory + --at\n")
                return 2
            err = axes.diff(programmed, result["vector"])
            data = {"programmed": programmed, "measured": result["vector"], **err}
            if args.at:
                data["at"] = args.at
            out = F.feature_frame(data, role=f"vector-diff:{role}", of=(src or {}).get("id"),
                                  lineage=[src["id"]] if src and src.get("id") else None,
                                  op="vector-diff", op_version="vector-diff@1", params={"sr_hz": sr})
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"vox vector: {exc}\n")
        ndjson.write_frames(list(frames) + [error_frame("op_failed", str(exc),
                            of=(src or {}).get("id"), op="vector")])
        sys.stdout.buffer.flush()
        return 1

    ndjson.write_frames(list(frames) + [out])
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
