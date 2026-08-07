"""The body registry — durable, named carrier voices with measured fingerprints.

A *body* is a named coordinate in carrier-voice space: an engine + params that render a
raw voice-like tone, optionally tagged (free-form ``tags:``) and carrying a measured
FINGERPRINT (f0 / HNR / inharmonicity / spectral shape). The palette (``bodies.yaml``)
outlives the engines under it — the same way a take-card outlives its renderer. This
module loads + validates the palette, renders any body to audio, and measures
fingerprints back into the yaml.

Engines:
  - ``sc-nrt``        — a shipped SuperCollider SynthDef (``params.synthdef`` names it,
                        resolved from vox-core package data) rendered offline via
                        smpl_synth.backends.render_nrt. Needs sclang/scsynth on PATH.
  - ``fof``           — the voxFof FOF/chant voice, rendered via the same SC NRT path with
                        the entry's formant params (a named sub-case of sc-nrt).
  - ``larynx-recipe`` — a WORLD re-voicing of a source clip (``params.source``) via
                        vox-larynx — no SuperCollider needed.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from vox_core import measure_f0_guarded, synthdef_path

# tools/vox-bodies (this file: tools/vox-bodies/src/vox_bodies/registry.py).
_TOOL_ROOT = Path(__file__).resolve().parents[2]
BODIES_YAML = _TOOL_ROOT / "bodies.yaml"

VALID_ENGINES = {"sc-nrt", "larynx-recipe", "fof"}
_REQUIRED_FIELDS = ("name", "engine")


# ---------------------------------------------------------------------------
# Registry load + validation.
# ---------------------------------------------------------------------------
def load_bodies(path: str | Path | None = None) -> list[dict]:
    """Load + validate the body palette. Returns the list of body dicts.

    Validates: ``name`` / ``engine`` present on every entry; ``engine`` is one of
    :data:`VALID_ENGINES`; ``tags`` (when present) is a list; names are unique. Raises
    ``ValueError`` on any violation (a malformed palette is a hard error, never a silent skip).
    """
    import yaml

    p = Path(path).expanduser() if path else BODIES_YAML
    data = yaml.safe_load(p.read_text()) or {}
    bodies = data.get("bodies") if isinstance(data, dict) else data
    if not isinstance(bodies, list):
        raise ValueError(f"bodies palette must be a list (or {{bodies: [...]}}); got {type(bodies).__name__}")

    seen: set[str] = set()
    for i, body in enumerate(bodies):
        if not isinstance(body, dict):
            raise ValueError(f"body #{i} is not a mapping: {body!r}")
        for field in _REQUIRED_FIELDS:
            if not body.get(field):
                raise ValueError(f"body #{i} missing required field {field!r}: {body!r}")
        name, engine = body["name"], body["engine"]
        if engine not in VALID_ENGINES:
            raise ValueError(f"body {name!r}: unknown engine {engine!r} (one of {sorted(VALID_ENGINES)})")
        tags = body.get("tags")
        if tags is not None and not isinstance(tags, list):
            raise ValueError(f"body {name!r}: tags must be a list, got {type(tags).__name__}")
        if name in seen:
            raise ValueError(f"duplicate body name {name!r}")
        seen.add(name)
    return bodies


def get_body(name: str, path: str | Path | None = None) -> dict:
    """Return one body entry by name (raises ``KeyError`` if absent)."""
    for body in load_bodies(path):
        if body["name"] == name:
            return body
    raise KeyError(f"no body named {name!r} in palette")


# ---------------------------------------------------------------------------
# Render.
# ---------------------------------------------------------------------------
def render_body(name: str, dur: float = 2.0, sr: int = 44100, path: str | Path | None = None):
    """Render a body to ``(samples float64 mono, sr)``.

    ``dur`` sets the note length for the SC engines (sc-nrt / fof). The larynx-recipe engine
    renders its source clip end-to-end, so its output length + sample rate come from the source
    (``dur`` / ``sr`` are advisory there).
    """
    body = get_body(name, path)
    engine = body["engine"]
    params = dict(body.get("params") or {})

    if engine == "larynx-recipe":
        return _render_larynx(params)
    if engine == "fof":
        return _render_sc("voxFof", params, dur, sr)
    if engine == "sc-nrt":
        synthdef = params.get("synthdef")
        if not synthdef:
            raise ValueError(f"body {name!r}: sc-nrt engine requires params.synthdef")
        return _render_sc(synthdef, params, dur, sr)
    raise ValueError(f"body {name!r}: unrenderable engine {engine!r}")


def _render_sc(synthdef_name: str, params: dict, dur: float, sr: int):
    """Render a shipped SC SynthDef offline via the smpl-synth NRT bridge; mono float64 + sr."""
    import soundfile as sf
    from smpl_synth.backends import render_nrt

    source = synthdef_path(synthdef_name).read_text()

    # `synthdef` is registry metadata (which .scd), not a SynthDef arg — strip it before the call.
    call_params = {k: v for k, v in params.items() if k != "synthdef"}
    call_params.setdefault("amp", 0.3)
    call_params["dur"] = float(dur)  # drive the SynthDef's own linen envelope

    wav = render_nrt(
        synthdef_source=source,
        synth_name=synthdef_name,
        params=call_params,
        duration=float(dur),
        sr=int(sr),
    )
    y, out_sr = sf.read(io.BytesIO(wav), dtype="float64", always_2d=True)
    return y.mean(axis=1), int(out_sr)


def _render_larynx(params: dict):
    """WORLD re-voice a source clip (no SuperCollider); return mono float64 + the source's sr."""
    import soundfile as sf
    from vox_larynx import world

    src = params.get("source")
    if not src:
        raise ValueError("larynx-recipe body requires params.source")
    path = Path(str(src)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"larynx-recipe source clip not found: {path}")
    data, in_sr = sf.read(str(path), dtype="float64", always_2d=True)
    x = data.mean(axis=1)
    y, _meta = world.render(
        x, int(in_sr),
        to_hz=params.get("to_hz"),
        formant_ratio=float(params.get("formant_ratio", 1.0)),
    )
    return y, int(in_sr)


# ---------------------------------------------------------------------------
# Fingerprint measurement.
# ---------------------------------------------------------------------------
def _spectral_shape(x: np.ndarray, sr: int):
    """``(centroid_hz, rolloff95_hz)`` — power-weighted centroid + the 95%-cumulative-energy freq.

    Cheap single-FFT descriptors that expose dead vocoder bands: a 2.5 kHz-LPF'd growl caps its
    rolloff near the cutoff, and a formant-locked body shows a low centroid. Returns (None, None)
    on too-short input.
    """
    n = len(x)
    if n < 512:
        return None, None
    xf = np.ascontiguousarray(x, dtype="float64")
    spec = np.abs(np.fft.rfft(xf * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total = float(np.sum(spec))
    if total <= 0:
        return None, None
    centroid = float(np.sum(freqs * spec) / total)
    cumulative = np.cumsum(spec)
    idx = int(np.searchsorted(cumulative, 0.95 * total))
    idx = min(idx, freqs.size - 1)
    rolloff = float(freqs[idx])
    return centroid, rolloff


def measure_body(y: np.ndarray, sr: int, target_hint: float | None = None) -> dict:
    """Fingerprint one rendered body.

    Returns ``{f0_hz, hnr_db, inharmonicity, centroid_hz, rolloff_hz}`` (any field nullable).

    f0 goes through vox-core's ``measure_f0_guarded`` (bass-safe ruler; ``target_hint`` is the
    authored target pitch, so a sub-90 Hz carrier can't silently octave-jump via parselmouth).
    HNR is parselmouth's harmonics-to-noise ratio (null when it reads a body as unvoiced).
    Inharmonicity is recomputed at the CHOSEN f0 so harsh FM / unreal-formant bodies get a real
    value instead of the describe fallback. centroid_hz / rolloff_hz expose dead vocoder bands.
    """
    from vox_ear import descriptors

    xf = np.ascontiguousarray(y, dtype="float64")
    d = descriptors.describe(xf, int(sr))

    guard = measure_f0_guarded(xf, int(sr), target_hint=target_hint)
    f0 = guard["f0_hz"]
    inharm = descriptors._inharmonicity(xf, int(sr), f0) if f0 else d.get("voice.inharmonicity")
    centroid, rolloff = _spectral_shape(xf, int(sr))
    return {
        "f0_hz": (round(f0, 2) if f0 is not None else None),
        "hnr_db": d.get("voice.hnr_db"),
        "inharmonicity": (round(float(inharm), 4) if inharm is not None else None),
        "centroid_hz": (round(centroid, 1) if centroid is not None else None),
        "rolloff_hz": (round(rolloff, 1) if rolloff is not None else None),
    }


def measure_fingerprints(path: str | Path | None = None, dur: float = 2.0, write: bool = True) -> dict:
    """Render every body (``dur`` s), measure its fingerprint, and (optionally) write it back.

    Returns ``{name: fingerprint}``. When ``write`` is True the palette yaml is rewritten with
    each entry's ``fingerprint`` populated (field order preserved; a header comment re-added).
    """
    import yaml

    p = Path(path).expanduser() if path else BODIES_YAML
    bodies = load_bodies(p)
    out: dict[str, dict] = {}
    for body in bodies:
        y, sr = render_body(body["name"], dur=dur, path=p)
        hint = (body.get("params") or {}).get("freq") or (body.get("params") or {}).get("to_hz")
        fp = measure_body(np.asarray(y), sr, target_hint=(float(hint) if hint else None))
        body["fingerprint"] = fp
        out[body["name"]] = fp
    if write:
        header = (
            "# bodies.yaml — the carrier-voice palette.\n"
            "# fingerprint.{f0_hz,hnr_db,inharmonicity,centroid_hz,rolloff_hz} are MEASURED by\n"
            "# registry.measure_fingerprints (f0 via vox-core's bass-safe guarded ruler); do not\n"
            "# hand-edit those — re-run the fingerprint pass instead.\n"
        )
        p.write_text(header + yaml.safe_dump({"bodies": bodies}, sort_keys=False))
    return out
