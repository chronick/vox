"""Resolve a cast spec to concrete model files.

A "cast" is a trained RVC voice model: a ``.pth`` weights file plus an
optional ``.index`` retrieval file, usually side by side in one directory
(the layout Applio exports). ``--model`` accepts three spellings:

- a path to the ``.pth`` file itself,
- a path to a directory holding exactly one ``.pth``,
- a bare name, looked up as a directory under the cast library
  (``~/.vox/casts`` by default, ``VOX_CASTS_DIR`` to override).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_CASTS_DIR = "~/.vox/casts"


class CastNotFound(ValueError):
    """Raised with a complete, printable message when a spec can't resolve."""


def casts_dir() -> Path:
    return Path(os.environ.get("VOX_CASTS_DIR", DEFAULT_CASTS_DIR)).expanduser()


def _pick_index(d: Path, override: str | None) -> Path | None:
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise CastNotFound(f"index file not found: {p}")
        return p
    found = sorted(d.glob("*.index"))
    if len(found) > 1:
        names = ", ".join(f.name for f in found)
        raise CastNotFound(f"multiple .index files in {d} ({names}); pass --index to pick one")
    return found[0] if found else None


def _from_dir(d: Path, index: str | None) -> dict:
    pths = sorted(d.glob("*.pth"))
    if not pths:
        raise CastNotFound(f"no .pth model file in {d}")
    if len(pths) > 1:
        names = ", ".join(f.name for f in pths)
        raise CastNotFound(
            f"multiple .pth files in {d} ({names}); pass the .pth path directly"
        )
    pth = pths[0]
    return {"name": pth.stem, "dir": d, "pth": pth, "index": _pick_index(d, index)}


def resolve_cast(spec: str, index: str | None = None) -> dict:
    """Resolve ``spec`` to ``{name, dir, pth, index}`` or raise CastNotFound."""
    p = Path(spec).expanduser()
    if p.is_file():
        if p.suffix != ".pth":
            raise CastNotFound(f"{p} is not a .pth model file")
        return {"name": p.stem, "dir": p.parent, "pth": p, "index": _pick_index(p.parent, index)}
    if p.is_dir():
        return _from_dir(p, index)
    if os.sep not in spec:
        lib = casts_dir() / spec
        if lib.is_dir():
            return _from_dir(lib, index)
        available = sorted(x.name for x in casts_dir().glob("*") if x.is_dir())
        listing = f" — available: {', '.join(available)}" if available else ""
        raise CastNotFound(
            f"no cast named {spec!r} under {casts_dir()}{listing}"
        )
    raise CastNotFound(f"model path not found: {spec}")


def _mb(p: Path) -> float:
    return round(p.stat().st_size / 1e6, 1)


def cast_info(resolved: dict) -> dict:
    """Describe a resolved cast from what's on disk (no torch needed)."""
    info: dict = {
        "name": resolved["name"],
        "dir": str(resolved["dir"]),
        "pth": {"file": resolved["pth"].name, "mb": _mb(resolved["pth"])},
        "index": None,
    }
    if resolved["index"]:
        info["index"] = {"file": resolved["index"].name, "mb": _mb(resolved["index"])}
    cfg = resolved["dir"] / "config.json"
    if cfg.is_file():
        try:
            sr = json.loads(cfg.read_text()).get("data", {}).get("sample_rate")
            if sr:
                info["sample_rate"] = sr
        except (json.JSONDecodeError, OSError):
            pass
    mi = resolved["dir"] / "model_info.json"
    if mi.is_file():
        try:
            info["model_info"] = json.loads(mi.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return info
