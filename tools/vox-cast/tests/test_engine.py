"""Atomic engine readiness and pinned-download integrity (no live network)."""

from __future__ import annotations

import hashlib
import io

import pytest
from vox_cast import engine


class FakeResponse(io.BytesIO):
    def __init__(self, payload, *, content_length=None):
        super().__init__(payload)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)


def _partial_engine(tmp_path):
    root = tmp_path / "engine"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python").write_text("placeholder")
    package = root / "lib" / "rvc_python"
    base = package / "base_model"
    base.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for filename, expected in engine.BASE_MODELS.items():
        path = base / filename
        path.write_bytes(b"")
        with path.open("r+b") as fh:
            fh.truncate(expected["size"])
    return root, base


def test_partial_engine_without_completion_marker_is_not_installed(tmp_path):
    _root, _base = _partial_engine(tmp_path)

    assert engine.is_installed() is False
    assert "completion marker is missing" in engine.status()["problems"]


def test_completed_engine_requires_every_exact_sized_model(tmp_path):
    _root, base = _partial_engine(tmp_path)
    engine._write_completion_marker(base)

    assert engine.is_installed() is True
    assert engine.status()["base_models_mb"] == pytest.approx(732.4, abs=0.1)

    (base / "rmvpe.pt").unlink()
    assert engine.is_installed() is False
    assert any("rmvpe.pt" in problem for problem in engine.status()["problems"])


def test_pinned_download_verifies_bytes_without_live_network(tmp_path, monkeypatch):
    payload = b"fixture model bytes"
    expected = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    seen = {}

    def fake_urlopen(url, timeout):
        seen.update(url=url, timeout=timeout)
        return FakeResponse(payload, content_length=len(payload))

    monkeypatch.setattr(engine.request, "urlopen", fake_urlopen)
    engine._download_model(tmp_path, "fixture.pt", expected)

    assert (tmp_path / "fixture.pt").read_bytes() == payload
    assert engine.MODEL_REVISION in seen["url"]
    assert seen["timeout"] == 120


def test_pinned_download_rejects_mismatched_content_length_before_reading(
    tmp_path, monkeypatch,
):
    class UnreadableResponse(FakeResponse):
        def read(self, _size=-1):
            raise AssertionError("body must not be read after Content-Length mismatch")

    expected = {"size": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
    response = UnreadableResponse(b"evil", content_length=5)
    monkeypatch.setattr(engine.request, "urlopen", lambda _url, timeout: response)

    with pytest.raises(RuntimeError, match="Content-Length 5 does not match expected size 4"):
        engine._download_model(tmp_path, "fixture.pt", expected)

    assert not (tmp_path / "fixture.pt").exists()
    assert not any(path.name.endswith(".download") for path in tmp_path.iterdir())


def test_pinned_download_aborts_when_stream_exceeds_expected_size(tmp_path, monkeypatch):
    class ChunkedResponse:
        headers = {}

        def __init__(self):
            self.reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            self.reads += 1
            return {1: b"good", 2: b"!", 3: b"never"}.get(self.reads, b"")

    expected = {"size": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
    response = ChunkedResponse()
    monkeypatch.setattr(engine.request, "urlopen", lambda _url, timeout: response)

    with pytest.raises(RuntimeError, match="downloaded size exceeded expected 4 bytes"):
        engine._download_model(tmp_path, "fixture.pt", expected)

    assert response.reads == 2
    assert not (tmp_path / "fixture.pt").exists()
    assert not any(path.name.endswith(".download") for path in tmp_path.iterdir())


def test_bad_pinned_download_never_replaces_destination(tmp_path, monkeypatch):
    expected = {"size": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
    monkeypatch.setattr(engine.request, "urlopen", lambda _url, timeout: io.BytesIO(b"evil"))

    with pytest.raises(RuntimeError, match="SHA-256"):
        engine._download_model(tmp_path, "fixture.pt", expected)

    assert not (tmp_path / "fixture.pt").exists()
    assert not any(path.name.endswith(".download") for path in tmp_path.iterdir())
