"""`vox syllabank` CLI — build the seed, look up syllables, report stats.

    vox syllabank build-seed [--root R]         seed the bank from say + FOF self-renders
    vox syllabank lookup WORD [--f0 HZ]         rank bank entries for a word's first syllable
    vox syllabank stats [--root R]              per-source counts + provenance summary

Reached via PATH discovery (`vox syllabank`); this tool lives in its own isolated venv
because of the pyworld/pronouncing deps.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import bank as B
from . import seed as S


def _cmd_build_seed(args) -> int:
    wordlist = None
    if getattr(args, "wordlist", None):
        with open(args.wordlist) as fh:
            wordlist = [w.strip() for w in fh if w.strip() and not w.startswith("#")]
    summary = S.build_seed(root=args.root, wordlist=wordlist)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_lookup(args) -> int:
    phones, syllables = B.word_syllables(args.word)
    first = syllables[0] if syllables else []
    first = first or None  # empty (out-of-cmudict) → null phones
    try:
        bank = B.load_bank(args.root)
    except ValueError as exc:
        sys.stderr.write(f"syllabank: {exc}\n")
        return 1
    results = B.lookup(bank, first, f0_hint=args.f0, top=args.top)
    out = {
        "word": args.word,
        "phones": phones,
        "first_syllable": first,
        "results": [
            {"id": e["id"], "syllable": e.get("syllable"), "phones": e.get("phones"),
             "f0_hz": e.get("f0_hz"), "dur_s": e.get("dur_s"), "source": e.get("source"),
             "license": e.get("license"), "tier": tier}
            for e, tier in results
        ],
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_stats(args) -> int:
    try:
        bank = B.load_bank(args.root)
    except ValueError as exc:
        sys.stderr.write(f"syllabank: {exc}\n")
        return 1
    entries = bank["entries"]
    by_source: dict[str, int] = {}
    by_license: dict[str, int] = {}
    for e in entries:
        by_source[e.get("source")] = by_source.get(e.get("source"), 0) + 1
        by_license[e.get("license")] = by_license.get(e.get("license"), 0) + 1
    print(json.dumps({
        "root": bank["root"],
        "entries": len(entries),
        "by_source": dict(sorted(by_source.items())),
        "by_license": dict(sorted(by_license.items())),
    }, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vox syllabank",
                                description="syllable-addressable vocal sample bank")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-seed", help="seed the bank from say + FOF self-renders")
    b.add_argument("--root", help="bank root (default $VOX_BANK_DIR or ~/.vox/bank)")
    b.add_argument("--wordlist", help="newline-separated word file replacing the default list")

    lk = sub.add_parser("lookup", help="rank bank entries for a word's first syllable")
    lk.add_argument("word")
    lk.add_argument("--f0", type=float, default=None, help="f0 hint (Hz) for within-tier ranking")
    lk.add_argument("--top", type=int, default=5, help="max results (default 5)")
    lk.add_argument("--root", help="bank root (default $VOX_BANK_DIR or ~/.vox/bank)")

    st = sub.add_parser("stats", help="per-source + per-license entry counts")
    st.add_argument("--root", help="bank root (default $VOX_BANK_DIR or ~/.vox/bank)")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.cmd == "build-seed":
        return _cmd_build_seed(args)
    if args.cmd == "lookup":
        return _cmd_lookup(args)
    if args.cmd == "stats":
        return _cmd_stats(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
