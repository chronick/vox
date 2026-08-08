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

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib import request

DEFAULT_ENGINE_DIR = "~/.vox/engines/rvc"
ENGINE_PYTHON_VERSION = "3.10"
ENGINE_LOCK = Path(__file__).with_name("engine-requirements.lock")
MODEL_REPO = "daswer123/rvc_base"
MODEL_REVISION = "bbb6736b97a98df0a87fe3592c0a061c53f0a75f"
BASE_MODELS = {
    "hubert_base.pt": {
        "sha256": "f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96",
        "size": 189_507_909,
    },
    "rmvpe.onnx": {
        "sha256": "5370e71ac80af8b4b7c793d27efd51fd8bf962de3a7ede0766dac0befa3660fd",
        "size": 361_688_443,
    },
    "rmvpe.pt": {
        "sha256": "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
        "size": 181_184_272,
    },
}
MARKER_SCHEMA = 2
MARKER_NAME = ".vox-rvc-complete.json"

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
    ready, _problems, _base_dir = _readiness()
    return ready


def completion_marker() -> Path:
    return engine_dir() / MARKER_NAME


def _marker_payload(base_dir: Path) -> dict:
    relative = base_dir.resolve().relative_to(engine_dir().resolve()).as_posix()
    return {
        "schema": MARKER_SCHEMA,
        "source": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "engine_lock_sha256": _sha256(ENGINE_LOCK),
        "base_model_dir": relative,
        "models": BASE_MODELS,
    }


def _readiness() -> tuple[bool, list[str], Path | None]:
    """Cheap readiness check: require a completed manifest and exact artifact sizes.

    SHA-256 is checked before the marker is atomically published. Rechecking 732 MB
    on every conversion would make readiness itself a substantial startup cost, so
    steady-state probes validate the marker plus package/model presence and sizes.
    """
    problems = []
    if not engine_python().is_file():
        problems.append("engine Python is missing")
    marker_path = completion_marker()
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.append("completion marker is missing")
        return False, problems, None
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"completion marker is unreadable: {exc}")
        return False, problems, None

    if marker.get("schema") != MARKER_SCHEMA:
        problems.append("completion marker schema does not match")
    if marker.get("source") != {"repo": MODEL_REPO, "revision": MODEL_REVISION}:
        problems.append("base-model source revision does not match")
    try:
        lock_sha256 = _sha256(ENGINE_LOCK)
    except OSError:
        problems.append("packaged engine dependency lock is missing")
    else:
        if marker.get("engine_lock_sha256") != lock_sha256:
            problems.append("engine dependency lock does not match")
    if marker.get("models") != BASE_MODELS:
        problems.append("base-model manifest does not match")

    relative = marker.get("base_model_dir")
    base_dir = None
    if not isinstance(relative, str):
        problems.append("completion marker has no base-model directory")
    else:
        rel_path = Path(relative)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            problems.append("completion marker has an unsafe base-model directory")
        else:
            base_dir = engine_dir() / rel_path
            if not (base_dir.parent / "__init__.py").is_file():
                problems.append("rvc_python package is missing")
            for filename, expected in BASE_MODELS.items():
                path = base_dir / filename
                try:
                    size = path.stat().st_size
                except OSError:
                    problems.append(f"base model is missing: {filename}")
                    continue
                if size != expected["size"]:
                    problems.append(
                        f"base model size mismatch: {filename} ({size} != {expected['size']})"
                    )
    return not problems, problems, base_dir


def _runner_path() -> Path:
    return Path(__file__).parent / "runner.py"


# Runs inside the engine venv: report without importing rvc_python (whose
# import drags in torch — seconds of startup for a status line).
_STATUS_SNIPPET = """\
import importlib.metadata as md
import json
import sys

out = {"python": sys.version.split()[0]}
try:
    out["rvc_python"] = md.version("rvc-python")
except md.PackageNotFoundError:
    out["rvc_python"] = None
print(json.dumps(out))
"""

_BASE_DIR_SNIPPET = """\
import importlib.util
import pathlib
spec = importlib.util.find_spec("rvc_python")
if not spec or not spec.origin:
    raise SystemExit("rvc_python package not found")
print(pathlib.Path(spec.origin).parent / "base_model")
"""


def status() -> dict:
    ready, problems, base_dir = _readiness()
    st: dict = {"dir": str(engine_dir()), "installed": ready}
    if base_dir is not None:
        sizes = {}
        for filename in BASE_MODELS:
            path = base_dir / filename
            if path.is_file():
                sizes[filename] = round(path.stat().st_size / 1e6, 1)
        st["base_models"] = sizes
        st["base_models_mb"] = round(
            sum((base_dir / filename).stat().st_size
                for filename in BASE_MODELS if (base_dir / filename).is_file()) / 1e6,
            1,
        )
    if not st["installed"]:
        st["problems"] = problems
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


def _probe_base_model_dir() -> Path:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    r = subprocess.run(
        [str(engine_python()), "-c", _BASE_DIR_SNIPPET],
        capture_output=True, text=True, env=env, check=False,
    )
    if r.returncode != 0 or not r.stdout.strip():
        detail = (r.stderr or r.stdout or "rvc_python package probe failed").strip()
        raise RuntimeError(detail[-500:])
    return Path(r.stdout.strip().splitlines()[-1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model(path: Path, expected: dict, *, checksum: bool) -> tuple[bool, str | None]:
    try:
        size = path.stat().st_size
    except OSError:
        return False, "file is missing"
    if size != expected["size"]:
        return False, f"size {size} != {expected['size']}"
    if checksum:
        actual = _sha256(path)
        if actual != expected["sha256"]:
            return False, f"SHA-256 {actual} != {expected['sha256']}"
    return True, None


def _download_model(base_dir: Path, filename: str, expected: dict) -> None:
    destination = base_dir / filename
    valid, _reason = _verify_model(destination, expected, checksum=True)
    if valid:
        print(f"verified cached {filename}", flush=True)
        return

    url = (
        f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{filename}"
        "?download=true"
    )
    print(f"downloading pinned {filename} ({expected['size'] / 1e6:.1f} MB) …", flush=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{filename}.", suffix=".download",
            dir=base_dir, delete=False,
        ) as fh:
            temp_path = Path(fh.name)
            digest = hashlib.sha256()
            total = 0
            with request.urlopen(url, timeout=120) as response:
                headers = getattr(response, "headers", None)
                declared_length = headers.get("Content-Length") if headers is not None else None
                if declared_length is not None:
                    try:
                        declared_size = int(declared_length)
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"{filename}: invalid Content-Length {declared_length!r}"
                        ) from exc
                    if declared_size != expected["size"]:
                        raise RuntimeError(
                            f"{filename}: Content-Length {declared_size} does not match "
                            f"expected size {expected['size']}"
                        )
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > expected["size"]:
                        raise RuntimeError(
                            f"{filename}: downloaded size exceeded expected "
                            f"{expected['size']} bytes"
                        )
                    fh.write(chunk)
                    digest.update(chunk)
            fh.flush()
            os.fsync(fh.fileno())
        if total != expected["size"]:
            raise RuntimeError(
                f"{filename}: downloaded size {total} does not match {expected['size']}"
            )
        actual = digest.hexdigest()
        if actual != expected["sha256"]:
            raise RuntimeError(
                f"{filename}: downloaded SHA-256 {actual} does not match {expected['sha256']}"
            )
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _write_completion_marker(base_dir: Path) -> None:
    payload = _marker_payload(base_dir)
    d = engine_dir()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{MARKER_NAME}.",
            dir=d, delete=False,
        ) as fh:
            temp_path = Path(fh.name)
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, completion_marker())
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def convert_env() -> dict:
    env = dict(os.environ)
    if sys.platform == "darwin":
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def setup(fresh: bool = False) -> int:
    """Build the engine, verify pinned base models, then publish readiness atomically."""
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
    # Any failure below leaves the environment inspectable/resumable, but never
    # advertises a partial package install or model download as conversion-ready.
    completion_marker().unlink(missing_ok=True)
    if not ENGINE_LOCK.is_file():
        sys.stderr.write(f"vox cast: packaged engine lock is missing: {ENGINE_LOCK}\n")
        return 1
    print("synchronizing the fully locked ML stack (rvc-python + torch) …", flush=True)
    r = subprocess.run(
        [uv, "pip", "sync", "--python", str(engine_python()),
         "--require-hashes", str(ENGINE_LOCK)],
        check=False,
    )
    if r.returncode != 0:
        return r.returncode
    print("verifying pinned base models (hubert + rmvpe, ~732 MB total) …", flush=True)
    try:
        base_dir = _probe_base_model_dir()
        for filename, expected in BASE_MODELS.items():
            _download_model(base_dir, filename, expected)
        _write_completion_marker(base_dir)
    except (OSError, RuntimeError) as exc:
        sys.stderr.write(f"vox cast: base-model setup failed: {exc}\n")
        return 1
    if not is_installed():
        sys.stderr.write("vox cast: engine setup completed but readiness verification failed\n")
        return 1
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
