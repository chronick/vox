"""Contract test: the REAL runner.py, executed against the stub engine packages.

This is the code that runs inside the engine venv; here it runs under the
test interpreter with stub torch/rvc_python on PYTHONPATH, so argument
plumbing, the MPS door-close, set_params mapping, and output checking are
all exercised for real.
"""

import json
import os
import subprocess
import sys

from conftest import RUNNER, make_wav


def _run_runner(tmp_path, stub_site, extra_args=()):
    in_wav = make_wav(tmp_path / "in.wav")
    out_wav = tmp_path / "out.wav"
    record = tmp_path / "record.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(stub_site)
    env["FAKE_RVC_RECORD"] = str(record)
    r = subprocess.run(
        [sys.executable, str(RUNNER), str(in_wav), str(out_wav),
         "--pth", str(tmp_path / "m.pth"), *extra_args],
        capture_output=True, text=True, env=env, check=False,
    )
    return r, out_wav, record


def test_runner_converts_and_reports(tmp_path, stub_site):
    r, out_wav, record = _run_runner(
        tmp_path, stub_site,
        ["--pitch", "2", "--f0-method", "harvest", "--index-rate", "0.7",
         "--protect", "0.2", "--rms-mix", "0.5"],
    )
    assert r.returncode == 0, r.stderr
    assert out_wav.stat().st_size > 0
    assert "seconds" in json.loads(r.stdout.strip().splitlines()[-1])
    rec = json.loads(record.read_text())
    assert rec["params"] == {
        "f0method": "harvest", "f0up_key": 2, "index_rate": 0.7,
        "protect": 0.2, "rms_mix_rate": 0.5,
    }
    assert rec["version"] == "v2"


def test_runner_closes_mps_door_on_cpu(tmp_path, stub_site):
    r, _out, record = _run_runner(tmp_path, stub_site)
    assert r.returncode == 0, r.stderr
    # Stub torch reports MPS available; on the default cpu device the runner
    # must have patched that to False before rvc-python could see it.
    assert json.loads(record.read_text())["mps_available_seen"] is False


def test_runner_leaves_mps_open_when_asked(tmp_path, stub_site):
    r, _out, record = _run_runner(tmp_path, stub_site, ["--device", "mps"])
    assert r.returncode == 0, r.stderr
    assert json.loads(record.read_text())["mps_available_seen"] is True
