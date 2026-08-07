"""Take-card orchestration — render, measure, verify against the programmed target.

A take-card is the durable artifact: it names a *coordinate in axis space* the take
should land on, plus how to render it. The renderer maps that onto whatever engines
exist today; the card outlives them. This module renders (larynx harmonize / larynx
render / flow), measures (ear → vector), and reports the programmed-vs-measured diff —
the self-verifying number.

Take-card schema (YAML):

    name: choir-probe
    source: /path/to/lead.wav          # required for larynx ops; omit for flow
    render:
      op: harmonize                    # harmonize | render | flow
      params: {chord: [0, 3, 7], drone: true, mode: follow}
    target: {breathiness: 0.15, roughness: 0.1, spatiality: 0.3}   # programmed coordinate (optional)
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from vox_ear import descriptors
from vox_flow import flow as flowmod
from vox_flow import render as flowrender
from vox_larynx import world
from vox_vector import axes


def _load_mono(path):
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    return data.mean(axis=1), int(sr)


def render_take(card: dict):
    """Execute a take-card's render op → ``(samples float32 mono, sr, render_meta)``."""
    render = card.get("render", {})
    op = render.get("op", "harmonize")
    params = dict(render.get("params", {}))

    if op == "flow":
        pattern = params.pop("pattern", "X.x.X.x.")
        syllables = params.pop("syllables", ["da", "ka", "ta", "ma"])
        score = flowmod.compile_flow(pattern, bpm=params.pop("bpm", 140),
                                     grid=params.pop("grid", 4), swing=params.pop("swing", 0.0),
                                     push_ms=params.pop("push_ms", 0.0), syllables=syllables)
        y = flowrender.render_flow(score, voice=params.pop("voice", "Fred"),
                                   pbas=params.pop("pbas", 92.0), sr=44100)
        chain = params.pop("chain", "grit")
        if chain != "none":
            y = flowrender.apply_chain(y, 44100, chain)
        return y.astype("float32"), 44100, {"op": "flow", "pattern": pattern, "n_onsets": score["n_onsets"]}

    src = card.get("source")
    if not src:
        raise ValueError(f"render op {op!r} requires a `source:` clip")
    x, sr = _load_mono(src)
    if op == "harmonize":
        chord = tuple(params.get("chord", (0, 3, 7)))
        y, meta = world.harmonize(x, sr, chord=chord, mode=params.get("mode", "follow"),
                                  include_lead=params.get("include_lead", True),
                                  drone=params.get("drone", False), seed=params.get("seed", 0))
        return y.astype("float32"), sr, {"op": "harmonize", **{k: meta[k] for k in ("chord", "mode", "voices")}}
    if op == "render":
        y, meta = world.render(x, sr, semitones=params.get("semitones", 0.0),
                               to_hz=params.get("to_hz"), flatten=params.get("flatten", False),
                               formant_ratio=params.get("formant_shift", 1.0),
                               time_ratio=params.get("stretch", 1.0))
        return y.astype("float32"), sr, {"op": "render", **meta}
    raise ValueError(f"unknown render op {op!r} (harmonize|render|flow)")


def measure_take(y, sr) -> dict:
    """ear descriptors → axis coordinate for a rendered take."""
    voice = descriptors.describe(np.ascontiguousarray(y, dtype="float64"), sr)
    vec = axes.measure(np.ascontiguousarray(y, dtype="float64"), sr, upstream=voice)
    return {"voice": voice, "vector": vec["vector"], "proxy_notes": vec["proxy_notes"]}


def run_take(card: dict) -> dict:
    """Render + measure + (optionally) verify a take-card. Returns the full report dict."""
    y, sr, render_meta = render_take(card)
    measured = measure_take(y, sr)
    report = {
        "name": card.get("name", "take"),
        "render": render_meta,
        "measured_vector": measured["vector"],
        "duration_s": round(len(y) / sr, 3),
        "audio": y, "sr": sr,
    }
    target = card.get("target")
    if target:
        diff = axes.diff(target, measured["vector"])
        report["target"] = target
        report["verification"] = diff
    return report
