"""Shipped-synthdef resolution tests.

    cd packages/vox-core && uv run pytest -q
"""

from __future__ import annotations

import pytest
from vox_core import sc


def test_all_shipped_synthdefs_resolve_and_parse():
    for name in sc.SYNTHDEFS:
        src = sc.synthdef_source(name)
        assert f"SynthDef(\\{name}" in src
        assert src.rstrip().endswith(".add;")


def test_unknown_synthdef_raises():
    with pytest.raises(KeyError):
        sc.synthdef_path("voxNope")
