"""`vox lyric` CLI — verify lyrics against a delivery spec, before any audio exists.

    vox lyric review --delivery percussive --lines "spit the code back|cut the deck to black"
    vox lyric review --spec verse.yaml                  → uses spec.delivery + spec.lines
    echo "one line per line" | vox lyric review --delivery sustained

Emits a `feature` frame (role ``lyric:<delivery>``) with the per-line review — blocklist
flags, choppability, delivery fit — plus a keep/rewrite verdict per line. The final taste
call stays with the writer; this is the pre-filter that catches the mechanical fails.
"""

from __future__ import annotations

import argparse
import io
import json
import sys

from . import packet, prosody


def _read_stdin_frames_or_text():
    if sys.stdin.isatty():
        return [], None
    data = sys.stdin.buffer.read()
    if not data.strip():
        return [], None
    stripped = data.lstrip()
    if stripped[:1] == b"{":
        from smplstream import ndjson
        return list(ndjson.read_frames(io.BytesIO(data))), None
    return [], data.decode("utf-8", "replace")


def _load_blocklist(path):
    if not path:
        return None
    with open(path) as fh:
        return frozenset(w.strip().lower() for w in fh if w.strip() and not w.startswith("#"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vox lyric", description="lyric prosody verifier")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review", help="verify lyrics against delivery prosody + gates")
    r.add_argument("--delivery", choices=prosody.DELIVERIES, help="required unless --spec sets it")
    r.add_argument("--lines", help="lyrics, '|'-separated (else --spec or stdin)")
    r.add_argument("--spec", help="spec YAML with `delivery:` and `lines:`")
    r.add_argument("--blocklist", help="newline-separated word file replacing the default blocklist")
    r.add_argument("--json", action="store_true", help="print the review as plain JSON (no frame)")

    k = sub.add_parser("packet", help="build the lyric packet (JSON for the score compiler)")
    k.add_argument("--delivery", choices=prosody.DELIVERIES, help="required unless --spec sets it")
    k.add_argument("--lines", help="lyrics, '|'-separated (else --spec or stdin)")
    k.add_argument("--spec", help="spec YAML with `delivery:` and `lines:`")
    k.add_argument("--blocklist", help="newline-separated word file replacing the default blocklist")

    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    frames, text = _read_stdin_frames_or_text()
    delivery, lines = args.delivery, None

    if args.spec:
        import yaml
        with open(args.spec) as fh:
            doc = yaml.safe_load(fh) or {}
        delivery = delivery or doc.get("delivery")
        lines = doc.get("lines")
    if args.lines:
        lines = args.lines.split("|")
    elif lines is None and text:
        lines = text.splitlines()

    if not delivery:
        sys.stderr.write("vox lyric: --delivery is required (or set `delivery:` in --spec)\n")
        return 2
    if not lines:
        sys.stderr.write("vox lyric: no lines (use --lines, --spec, or pipe text on stdin)\n")
        return 2

    blocklist = _load_blocklist(args.blocklist)

    if args.cmd == "packet":
        print(json.dumps(packet.build_packet(lines, delivery, blocklist), indent=2))
        return 0

    result = prosody.review(lines, delivery, blocklist)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    from smplstream import frames as F
    from smplstream import ndjson
    feat = F.feature_frame(result, role=f"lyric:{delivery}", op="lyric-review",
                           op_version="lyric-review@1",
                           params={"delivery": delivery, "n_lines": result["n_lines"]})
    ndjson.write_frames(list(frames) + [feat])
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
