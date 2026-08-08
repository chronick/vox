"""`vox tongue` CLI — compile words+melody into a TONGUE score, and sing a score to a WAV.

    vox tongue compile --lines "wake the machine" --melody "A2,C3,E3" --bpm 120 [--flow "X.x."] \
        [--score out.yaml]
        -> a validated TONGUE score (written to --score, else printed as YAML on stdout)

    vox lyric packet ... | vox tongue compile-packet --melody "A2,C3,E3" --score out.yaml
        -> compile reviewed packet phones + gates directly into a validated TONGUE score

    vox tongue sing --score s.yaml --out t.wav
        -> renders the score to t.wav via say->WORLD, and prints the per-syllable manifest JSON

The heavy dep is pyworld (a compiled wheel), so this tool lives in its own isolated <3.13 venv —
reached via `vox tongue`. Sibling tools vox_flow / vox_larynx are resolved on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import compile as compile_mod
from . import schema


def _split_lines(arg: str):
    # Lines may be separated by literal newlines or the '|' bar; fall back to a single line.
    parts = [p.strip() for p in arg.replace("|", "\n").splitlines()]
    return [p for p in parts if p] or [arg.strip()]


def _split_melody(arg: str):
    return [n.strip() for n in arg.split(",") if n.strip()]


def _cmd_compile(args) -> int:
    score = compile_mod.compile(
        _split_lines(args.lines),
        _split_melody(args.melody),
        bpm=args.bpm,
        flow_pattern=args.flow,
    )
    text = schema.to_yaml(score)
    if args.score:
        with open(args.score, "w", encoding="utf-8") as fh:
            fh.write(text)
        sys.stderr.write(f"wrote {args.score} ({len(score['syllables'])} syllables)\n")
    else:
        sys.stdout.write(text)
    return 0


def _load_packet(path: str | None) -> dict:
    if path and path != "-":
        with open(path, encoding="utf-8") as fh:
            packet = json.load(fh)
    else:
        packet = json.load(sys.stdin)
    if not isinstance(packet, dict):
        raise ValueError("packet JSON must be an object")
    return packet


def _cmd_compile_packet(args) -> int:
    try:
        packet = _load_packet(args.packet)
        score = compile_mod.compile_packet(
            packet,
            _split_melody(args.melody),
            bpm=args.bpm,
            flow_pattern=args.flow,
        )
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"vox tongue compile-packet: could not read lyric packet: {exc}\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"vox tongue compile-packet: lyric packet rejected: {exc}\n")
        return 2

    text = schema.to_yaml(score)
    if args.score:
        try:
            with open(args.score, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            sys.stderr.write(f"vox tongue compile-packet: could not write score: {exc}\n")
            return 1
        sys.stderr.write(f"wrote {args.score} ({len(score['syllables'])} syllables)\n")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_emit_ds(args) -> int:
    from . import (
        ds as ds_mod,  # lazy — pure-python, but keep the import local to the subcommand
    )

    score = schema.load_score(args.score)
    seg = ds_mod.to_ds(score, with_f0=not args.no_f0, f0_timestep=args.f0_timestep)
    text = json.dumps([seg], indent=2)  # a .ds file is a JSON array of segments
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        sys.stderr.write(
            f"wrote {args.out} ({len(seg['ph_seq'].split())} phones, "
            f"{len(seg['note_seq'].split())} notes)\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


def _cmd_sing(args) -> int:
    score = schema.load_score(args.score)

    if args.backend == "concat":
        from . import concat as concat_mod  # lazy — pulls in the syllabank sibling

        samples, manifest = concat_mod.render_concat(
            score, bank_root=args.bank, sr=args.sr, voice=args.voice)
    else:
        from . import (
            render as render_mod,  # lazy — keeps `compile` usable without pyworld/say
        )

        if not render_mod.say_available():
            sys.stderr.write("vox tongue: `say` not found on PATH — cannot render\n")
            return 1
        samples, manifest = render_mod.render(score, sr=args.sr, voice=args.voice)

    import soundfile as sf

    sf.write(args.out, samples, args.sr, subtype="FLOAT")
    sys.stderr.write(f"wrote {args.out} ({len(samples) / args.sr:.2f}s, backend={args.backend})\n")
    json.dump(manifest, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _cmd_warp(args) -> int:
    from . import warp as warp_mod  # lazy — pulls in faster-whisper + pyworld

    if not warp_mod.whisper_available():
        sys.stderr.write("vox tongue warp: faster-whisper not importable — cannot align\n")
        return 1

    import soundfile as sf

    score = schema.load_score(args.score)
    data, in_sr = sf.read(args.__dict__["in"], dtype="float64", always_2d=True)
    vocal = data.mean(axis=1)
    if in_sr != args.sr:  # 24 kHz Kokoro etc. -> the working rate (known trap: always resample up)
        from .concat import _resample

        vocal = _resample(vocal, int(in_sr), int(args.sr))

    samples, report = warp_mod.warp_to_score(vocal, args.sr, score, bpm=args.bpm)
    sf.write(args.out, samples, args.sr, subtype="FLOAT")
    sys.stderr.write(f"wrote {args.out} ({len(samples) / args.sr:.2f}s, "
                     f"{len(report['syllables'])} syllables)\n")
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _build_parser():
    p = argparse.ArgumentParser(prog="vox tongue", description="the TONGUE score DSL + render path")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="words + melody -> a TONGUE score")
    c.add_argument("--lines", required=True, help="lyric text (newline- or '|'-separated lines)")
    c.add_argument("--melody", required=True, help="comma notes/Hz, cycled (e.g. A2,C3,E3)")
    c.add_argument("--bpm", type=float, default=120.0)
    c.add_argument("--flow", help="optional FLOW grid pattern (e.g. 'X.x.') for authored placement")
    c.add_argument("--score", help="write the score YAML here (else printed to stdout)")
    c.set_defaults(func=_cmd_compile)

    cp = sub.add_parser(
        "compile-packet",
        help="reviewed vox-lyric packet JSON -> a TONGUE score",
    )
    cp.add_argument(
        "--packet",
        default="-",
        help="vox-lyric packet JSON file (default: stdin; use - for stdin)",
    )
    cp.add_argument("--melody", required=True, help="comma notes/Hz, cycled (e.g. A2,C3,E3)")
    cp.add_argument("--bpm", type=float, default=120.0)
    cp.add_argument("--flow", help="optional FLOW grid pattern overriding the packet flow hint")
    cp.add_argument("--score", help="write the score YAML here (else printed to stdout)")
    cp.set_defaults(func=_cmd_compile_packet)

    e = sub.add_parser("emit-ds", help="emit a DiffSinger .ds segment (JSON) from a TONGUE score")
    e.add_argument("--score", required=True, help="path to a TONGUE score YAML")
    e.add_argument("--out", help="write the .ds JSON here (else printed to stdout)")
    e.add_argument("--no-f0", action="store_true", help="omit the flat-per-note f0_seq curve")
    e.add_argument("--f0-timestep", type=float, default=0.024, help="f0 hop in seconds (default 0.024)")
    e.set_defaults(func=_cmd_emit_ds)

    s = sub.add_parser("sing", help="render a TONGUE score to a WAV (say->WORLD, or concat)")
    s.add_argument("--score", required=True, help="path to a TONGUE score YAML")
    s.add_argument("--out", required=True, help="output WAV path")
    s.add_argument("--backend", choices=("say", "concat"), default="say",
                   help="say = say+larynx synth (default); concat = stitch spliced syllabank clips")
    s.add_argument("--bank", help="syllabank root for --backend concat (default VOX_BANK_DIR)")
    s.add_argument("--voice", default="Fred", help="macOS `say` voice (default Fred)")
    s.add_argument("--sr", type=int, default=44100)
    s.set_defaults(func=_cmd_sing)

    w = sub.add_parser("warp", help="warp a sung clip onto a TONGUE score's grid (singing-warp)")
    w.add_argument("--in", required=True, help="input sung WAV (any sr; resampled to --sr)")
    w.add_argument("--score", required=True, help="path to a TONGUE score YAML (the grid target)")
    w.add_argument("--out", required=True, help="output warped WAV path")
    w.add_argument("--bpm", type=float, default=None, help="override the score's bpm")
    w.add_argument("--sr", type=int, default=44100, help="working/output sample rate")
    w.set_defaults(func=_cmd_warp)
    return p


def main(argv=None) -> int:
    try:
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, ValueError, AttributeError):
        pass

    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
