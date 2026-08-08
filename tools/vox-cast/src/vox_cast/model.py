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

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

DEFAULT_CASTS_DIR = "~/.vox/casts"


class CastNotFound(ValueError):
    """Raised with a complete, printable message when a spec can't resolve."""


class CastImportError(ValueError):
    """Raised when a local model cannot be copied safely into the cast library."""


_CAST_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SIDECAR_SUFFIXES = {
    ".json", ".yaml", ".yml", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
}
_SIDECAR_NAMES = {"LICENSE", "SHA256SUMS"}
_DIRECT_MODEL_SIDECAR_STEMS = {
    "config", "model_info", "MODEL_CARD", "LICENSE", "SHA256SUMS",
}


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
        return {
            "name": p.stem,
            "dir": p.parent,
            "pth": p,
            "index": _pick_index(p.parent, index),
            "direct_model": True,
        }
    if p.is_dir():
        return _from_dir(p, index)
    if os.sep not in spec:
        lib = casts_dir() / spec
        if lib.is_dir():
            resolved = _from_dir(lib, index)
            resolved["name"] = spec
            return resolved
        available = sorted(x.name for x in casts_dir().glob("*") if x.is_dir())
        listing = f" — available: {', '.join(available)}" if available else ""
        raise CastNotFound(
            f"no cast named {spec!r} under {casts_dir()}{listing}"
        )
    raise CastNotFound(f"model path not found: {spec}")


def _validate_name(name: str) -> str:
    if not _CAST_NAME.fullmatch(name) or name in {".", ".."}:
        raise CastImportError(
            "cast name must start with a letter or digit and contain only letters, "
            "digits, '.', '_', or '-'"
        )
    return name


def import_cast(spec: str, *, index: str | None = None, name: str | None = None) -> dict:
    """Copy a user-supplied local RVC model into ``VOX_CASTS_DIR``.

    Only the selected weights/index and common direct-child metadata/image/text
    sidecars are copied; training data and nested export artifacts stay put. The
    destination is assembled in a sibling temporary directory and published only
    after it resolves as a complete cast.
    """
    source = Path(spec).expanduser()
    if not source.exists():
        raise CastImportError(f"model path not found: {source}")
    try:
        resolved = resolve_cast(str(source), index=index)
    except CastNotFound as exc:
        raise CastImportError(str(exc)) from exc

    logical_name = _validate_name(name or (source.name if source.is_dir() else source.stem))
    library = casts_dir()
    destination = library / logical_name
    if destination.exists():
        raise CastImportError(
            f"cast {logical_name!r} already exists at {destination}; refusing to overwrite"
        )

    library.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{logical_name}.import-", dir=str(library)))
    try:
        copied: set[Path] = set()

        def copy_file(path: Path) -> None:
            src = path.resolve()
            if src in copied:
                return
            shutil.copy2(src, temp_dir / path.name)
            copied.add(src)

        copy_file(resolved["pth"])
        if resolved["index"] is not None:
            copy_file(resolved["index"])
        for candidate in sorted(resolved["dir"].iterdir()):
            if not candidate.is_file():
                continue
            recognized = (
                candidate.suffix.lower() in _SIDECAR_SUFFIXES
                or candidate.name in _SIDECAR_NAMES
            )
            if not recognized:
                continue
            # A directory is an intentional bundle, so preserve its recognized
            # direct-child sidecars. For a lone .pth, do not sweep unrelated
            # JSON/images from a broad folder such as ~/Downloads.
            if source.is_file() and not (
                candidate.stem == source.stem
                or candidate.stem in _DIRECT_MODEL_SIDECAR_STEMS
                or candidate.name in _SIDECAR_NAMES
            ):
                continue
            copy_file(candidate)

        imported = _from_dir(temp_dir, None)
        if destination.exists():
            raise CastImportError(
                f"cast {logical_name!r} appeared at {destination}; refusing to overwrite"
            )
        temp_dir.rename(destination)
        imported = _from_dir(destination, None)
        imported["name"] = logical_name
        return imported
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def list_casts() -> dict:
    """Describe every valid library cast and surface incomplete directories."""
    library = casts_dir()
    found = []
    invalid = []
    if library.is_dir():
        for directory in sorted(library.iterdir(), key=lambda p: p.name.lower()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            try:
                resolved = _from_dir(directory, None)
                resolved["name"] = directory.name
                found.append(cast_info(resolved))
            except CastNotFound as exc:
                invalid.append({"name": directory.name, "error": str(exc)})
    return {"dir": str(library), "casts": found, "invalid": invalid}


def _mb(p: Path) -> float:
    return round(p.stat().st_size / 1e6, 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_info(path: Path, *, checksum: bool) -> dict:
    info = {"file": path.name, "mb": _mb(path)}
    if checksum:
        info["sha256"] = _sha256(path)
    return info


def _sidecar_inventory(resolved: dict) -> list[dict]:
    artifacts = {resolved["pth"].resolve()}
    if resolved["index"]:
        artifacts.add(resolved["index"].resolve())
    inventory = []
    for candidate in sorted(resolved["dir"].iterdir(), key=lambda path: path.name.lower()):
        if not candidate.is_file() or candidate.resolve() in artifacts:
            continue
        recognized = (
            candidate.suffix.lower() in _SIDECAR_SUFFIXES
            or candidate.name in _SIDECAR_NAMES
        )
        if not recognized:
            continue
        if resolved.get("direct_model") and not (
            candidate.stem == resolved["pth"].stem
            or candidate.stem in _DIRECT_MODEL_SIDECAR_STEMS
            or candidate.name in _SIDECAR_NAMES
        ):
            continue
        inventory.append({"file": candidate.name, "mb": _mb(candidate)})
    return inventory


def cast_info(resolved: dict, *, checksums: bool = False) -> dict:
    """Describe a resolved cast from what's on disk (no torch needed)."""
    info: dict = {
        "name": resolved["name"],
        "dir": str(resolved["dir"]),
        "pth": _artifact_info(resolved["pth"], checksum=checksums),
        "index": None,
        "sidecars": _sidecar_inventory(resolved),
    }
    if resolved["index"]:
        info["index"] = _artifact_info(resolved["index"], checksum=checksums)
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
