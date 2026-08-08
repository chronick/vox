"""The RVC engine venv: a separate Python 3.10 environment holding the ML stack.

rvc-python's dependency pins (numpy<=1.23.5, fairseq==0.12.2) are
irreconcilable with the smpl/vox stack and with modern Pythons, so the
engine lives in its own venv that this module builds and invokes — never
imports. Conversion shells out to ``runner.py`` executed by the engine's
interpreter.

Version pins are load-bearing:

- Python 3.10 — fairseq 0.12.2 does not build on newer interpreters.
- torch==2.5.1 — torch>=2.6 flips ``torch.load`` to ``weights_only=True``,
  which breaks fairseq's hubert checkpoint load.
- setuptools<81 — pyworld imports ``pkg_resources`` at import time.

On macOS the engine subprocess runs with ``OMP_NUM_THREADS=1`` and
``KMP_DUPLICATE_LIB_OK=TRUE``: torch and faiss each bundle a libomp, and
loading both segfaults the process mid-inference without the workaround.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ENGINE_DIR = "~/.vox/engines/rvc"
ENGINE_PYTHON_VERSION = "3.10"
ENGINE_PACKAGES = (
    "rvc-python==0.1.5",
    "torch==2.5.1",
    "torchaudio==2.5.1",
    "setuptools<81",
)

INSTALL_HINT = (
    "run `vox cast setup` to build the engine "
    f"(isolated Python {ENGINE_PYTHON_VERSION} venv; ~3 GB once torch and the "
    "base models are in)"
)


def engine_dir() -> Path:
    return Path(os.environ.get("VOX_RVC_ENGINE", DEFAULT_ENGINE_DIR)).expanduser()


def engine_python() -> Path:
    return engine_dir() / "bin" / "python"


def is_installed() -> bool:
    return engine_python().exists()


def _runner_path() -> Path:
    return Path(__file__).parent / "runner.py"


# Runs inside the engine venv: report without importing rvc_python (whose
# import drags in torch — seconds of startup for a status line).
_STATUS_SNIPPET = """\
import importlib.metadata as md
import importlib.util
import json
import os
import sys

out = {"python": sys.version.split()[0]}
try:
    out["rvc_python"] = md.version("rvc-python")
except md.PackageNotFoundError:
    out["rvc_python"] = None
spec = importlib.util.find_spec("rvc_python")
if spec and spec.origin:
    base = os.path.join(os.path.dirname(spec.origin), "base_model")
    if os.path.isdir(base):
        out["base_models"] = {
            f: round(os.path.getsize(os.path.join(base, f)) / 1e6, 1)
            for f in sorted(os.listdir(base))
            if f.endswith((".pt", ".pth", ".onnx"))
        }
print(json.dumps(out))
"""

_PREDOWNLOAD_SNIPPET = """\
import os
import rvc_python
from rvc_python.download_model import download_rvc_models
download_rvc_models(os.path.dirname(rvc_python.__file__))
print("base models present")
"""


def status() -> dict:
    st: dict = {"dir": str(engine_dir()), "installed": is_installed()}
    if not st["installed"]:
        return st
    try:
        r = subprocess.run(
            [str(engine_python()), "-c", _STATUS_SNIPPET],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if r.returncode == 0:
            st.update(json.loads(r.stdout.strip().splitlines()[-1]))
        else:
            st["error"] = (r.stderr or "status probe failed").strip()[-500:]
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, IndexError) as exc:
        st["error"] = str(exc)
    return st


def convert_env() -> dict:
    env = dict(os.environ)
    if sys.platform == "darwin":
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def setup(fresh: bool = False) -> int:
    """Build (or rebuild) the engine venv with uv; pre-download base models."""
    uv = shutil.which("uv")
    if uv is None:
        sys.stderr.write("vox cast: `uv` not found on PATH — install it first "
                         "(https://docs.astral.sh/uv/)\n")
        return 1
    d = engine_dir()
    if fresh and d.exists():
        if not (d / "pyvenv.cfg").exists():
            sys.stderr.write(f"vox cast: {d} exists but is not a venv; refusing --fresh\n")
            return 1
        shutil.rmtree(d)
    if not engine_python().exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        print(f"creating engine venv (Python {ENGINE_PYTHON_VERSION}) at {d} …", flush=True)
        r = subprocess.run([uv, "venv", "--python", ENGINE_PYTHON_VERSION, str(d)], check=False)
        if r.returncode != 0:
            return r.returncode
    print("installing the ML stack (rvc-python + pinned torch) …", flush=True)
    r = subprocess.run(
        [uv, "pip", "install", "--python", str(engine_python()), *ENGINE_PACKAGES],
        check=False,
    )
    if r.returncode != 0:
        return r.returncode
    print("fetching base models (hubert + rmvpe, ~370 MB on first setup) …", flush=True)
    r = subprocess.run([str(engine_python()), "-c", _PREDOWNLOAD_SNIPPET],
                       env=convert_env(), check=False)
    if r.returncode != 0:
        return r.returncode
    print(json.dumps(status(), indent=2))
    return 0


def run_convert(in_wav: Path, out_wav: Path, *, pth: Path, index: Path | None,
                params: dict, cwd: Path) -> tuple[int, str]:
    """Convert ``in_wav`` → ``out_wav`` in the engine venv. Returns (rc, stderr)."""
    cmd = [
        str(engine_python()), str(_runner_path()),
        str(in_wav), str(out_wav),
        "--pth", str(pth),
        "--index", str(index) if index else "",
        "--pitch", str(params["pitch"]),
        "--f0-method", params["f0_method"],
        "--index-rate", str(params["index_rate"]),
        "--protect", str(params["protect"]),
        "--rms-mix", str(params["rms_mix"]),
        "--arch", params["arch"],
        "--device", params["device"],
    ]
    r = subprocess.run(cmd, cwd=str(cwd), env=convert_env(),
                       capture_output=True, text=True, check=False)
    return r.returncode, (r.stderr or "")


def rvc_version() -> str:
    if not is_installed():
        return "absent"
    st = status()
    return st.get("rvc_python") or "unknown"
