"""Shipped SuperCollider synthdefs, resolvable from the installed package.

The .scd sources live as package data so any vox tool can hand them to
smpl-synth's NRT bridge without a repo checkout.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

SYNTHDEFS = ("voxFof", "voxGrowl", "voxSubSaw", "voxThroat")


def synthdef_path(name: str) -> Path:
    """Filesystem path of a shipped synthdef (e.g. ``synthdef_path("voxFof")``)."""
    if name not in SYNTHDEFS:
        raise KeyError(f"unknown synthdef {name!r} (have: {', '.join(SYNTHDEFS)})")
    ref = resources.files("vox_core").joinpath(f"synthdefs/{name}.scd")
    with resources.as_file(ref) as p:
        return Path(p)


def synthdef_source(name: str) -> str:
    """Source text of a shipped synthdef."""
    return synthdef_path(name).read_text()
