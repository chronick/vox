"""Shared fixtures: isolated CAS, a fixture cast tree, and a fake engine.

The fake engine is the real trick: a venv-shaped directory whose
``bin/python`` is a symlink to the test interpreter, plus stub ``torch``
and ``rvc_python`` packages on PYTHONPATH. The REAL runner.py then runs
end to end — argument parsing, device patching, set_params, infer_file —
with the stub recording what it saw and copying input to output. No ML
stack, no network, still the true engine-side code path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from vox_cast import engine

SRC = Path(__file__).parent.parent / "src"
RUNNER = SRC / "vox_cast" / "runner.py"


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.setenv("VOX_CASTS_DIR", str(tmp_path / "casts"))
    monkeypatch.setenv("VOX_RVC_ENGINE", str(tmp_path / "engine"))


@pytest.fixture
def cast_tree(tmp_path):
    """A plausible Applio-export model dir: one .pth, one .index, metadata."""
    d = tmp_path / "casts" / "growl"
    d.mkdir(parents=True)
    (d / "growl.pth").write_bytes(b"\x00" * 2048)
    (d / "growl.index").write_bytes(b"\x00" * 1024)
    (d / "config.json").write_text(json.dumps({"data": {"sample_rate": 40000}}))
    (d / "model_info.json").write_text(json.dumps({"embedder_model": "contentvec"}))
    return d


STUB_TORCH = """\
class _MPS:
    @staticmethod
    def is_available():
        return True


class _Backends:
    def __init__(self):
        self.mps = _MPS()


backends = _Backends()
"""

# Mirrors the rvc_python.infer surface runner.py touches. Records every call
# into $FAKE_RVC_RECORD and "converts" by copying input to output.
STUB_RVC = """\
import json
import os
import shutil

import torch


class RVCInference:
    def __init__(self, device="cpu:0", model_path=None, index_path="", version="v2"):
        self._rec = {
            "device": device, "model_path": model_path,
            "index_path": index_path, "version": version,
            "mps_available_seen": bool(torch.backends.mps.is_available()),
        }

    def set_params(self, **kw):
        self._rec["params"] = kw

    def infer_file(self, input_path, output_path):
        shutil.copyfile(input_path, output_path)
        with open(os.environ["FAKE_RVC_RECORD"], "w") as f:
            json.dump(self._rec, f)
"""


@pytest.fixture
def stub_site(tmp_path):
    """A PYTHONPATH dir with stub torch + rvc_python packages."""
    site = tmp_path / "stub-site"
    (site / "rvc_python").mkdir(parents=True)
    (site / "torch").mkdir()
    (site / "torch" / "__init__.py").write_text(STUB_TORCH)
    (site / "rvc_python" / "__init__.py").write_text("")
    (site / "rvc_python" / "infer.py").write_text(STUB_RVC)
    return site


@pytest.fixture
def fake_engine(tmp_path, stub_site, monkeypatch):
    """Engine dir whose python is the test interpreter + stub PYTHONPATH.

    An exec shim, not a symlink: a venv interpreter invoked through a symlink
    at a foreign path can't find its own pyvenv.cfg and dies before `import
    encodings`; exec'ing it by its true path keeps it healthy.
    """
    d = tmp_path / "engine"
    (d / "bin").mkdir(parents=True)
    (d / "pyvenv.cfg").write_text("home = fake\n")
    shim = d / "bin" / "python"
    shim.write_text(f"#!/bin/sh\nexec \"{sys.executable}\" \"$@\"\n")
    shim.chmod(0o755)
    package = d / "lib" / "rvc_python"
    base_models = package / "base_model"
    base_models.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    # Readiness checks exact artifact sizes but does not rehash 732 MB on each
    # conversion. Sparse fixture files keep this hermetic and essentially free.
    for filename, expected in engine.BASE_MODELS.items():
        path = base_models / filename
        path.write_bytes(b"")
        with path.open("r+b") as fh:
            fh.truncate(expected["size"])
    engine._write_completion_marker(base_models)
    record = tmp_path / "rvc-record.json"
    monkeypatch.setenv("PYTHONPATH", str(stub_site))
    monkeypatch.setenv("FAKE_RVC_RECORD", str(record))
    return {"dir": d, "record": record}


def run_cli(args, stdin_bytes=b"", env_extra=None):
    """Run the CLI in a subprocess so real stdio/frames flow end to end."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(SRC), env.get("PYTHONPATH", "")) if p
    )
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c",
         "from vox_cast.cli import main; raise SystemExit(main())", *args],
        input=stdin_bytes, capture_output=True, env=env, check=False,
    )


def make_wav(path: Path, seconds=0.2, sr=22050):
    import numpy as np
    import soundfile as sf

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.1 * np.sin(2 * 3.14159 * 220 * t)).astype("float32"),
             sr, subtype="FLOAT")
    return path


def frames_bytes(wav: Path):
    """CAS the wav and serialize a one-frame NDJSON stream (an upstream `smpl read`)."""
    import io

    import soundfile as sf
    from smplstream import cas, ndjson
    from smplstream import frames as F

    data, sr = sf.read(str(wav), dtype="float64", always_2d=True)
    buf = io.BytesIO()
    sf.write(buf, data.astype("float32"), sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    fr = F.audio_frame(h, sr=sr, ch=1, dur=data.shape[0] / sr, role="voice",
                       op="read", op_version="test@1", params={})
    out = io.BytesIO()
    ndjson.write_frames([fr], out)
    return out.getvalue(), fr


def parse_frames(stdout_bytes):
    import io

    from smplstream import ndjson
    return list(ndjson.read_frames(io.BytesIO(stdout_bytes)))
