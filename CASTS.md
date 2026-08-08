# Voice casts (RVC models)

A **cast** is a trained voice: an RVC voice-conversion model that
re-voices any take you pour through it while keeping the performance —
the pitch contour, the timing, the words. `vox cast` loads casts and
converts audio through them as an smpl pipe stage.

**Training is out of scope by design.** vox loads models; it does not
make them. Train with any tool in the RVC ecosystem (Applio is the
usual choice, locally or on a rented GPU) and point `vox cast` at the
export. Any standard RVC v2 export works; v1 works with `--arch v1`.

## The one-time engine build

The ML stack behind RVC (rvc-python, torch, fairseq) pins dependencies
that cannot share an environment with the rest of vox — so it lives in
its own venv that the tool builds and shells out to:

```bash
vox cast setup            # builds ~/.vox/engines/rvc (~3 GB all-in)
vox cast setup --status   # report what's installed, change nothing
```

Setup needs `uv` and the network: it creates a Python 3.10 venv,
installs the pinned stack, and pre-downloads the shared base models
(hubert + rmvpe, ~370 MB — used by every cast). After that, conversion
is fully local and offline. `VOX_RVC_ENGINE` overrides the venv
location; `--fresh` rebuilds it.

The pins are deliberate, and each one is load-bearing; the story is in
`tools/vox-cast/src/vox_cast/engine.py`. Don't upgrade the engine venv
by hand.

## Using a cast

A cast on disk is a directory with one `.pth` (the weights, ~55 MB)
and usually one `.index` (the retrieval bank, often much larger):

```text
~/.vox/casts/mycast/
├── mycast.pth
├── mycast.index
├── config.json        # optional, from training
└── model_info.json    # optional, from training
```

`--model` takes a bare name from that library (`VOX_CASTS_DIR`
overrides `~/.vox/casts`), a directory, or a `.pth` path directly:

```bash
# describe a cast from its files (no ML stack loaded)
vox cast info --model mycast

# the pipe stage
smpl read take.wav | vox cast convert --model mycast | smpl write voiced.wav

# standalone, plus the knobs that matter
vox cast convert --in take.wav --model mycast --pitch 12 --index-rate 0.7
```

The output frame carries the conversion as `role: …wet`, with the
model name and every knob recorded in `params`, and comes out at the
cast's native sample rate (40 kHz for a standard export).

## The knobs

| Flag | Default | What it does |
|---|---|---|
| `--pitch` | 0 | transpose in semitones before conversion (+12: low source into a high cast) |
| `--index-rate` | 0.5 | 0–1, how hard to pull toward the cast's timbre bank (higher = more of the cast, less of the room) |
| `--protect` | 0.33 | 0–0.5, guards consonants and breaths from conversion artifacts (lower = more conversion) |
| `--f0-method` | rmvpe | pitch tracker; `rmvpe` is the robust default, `harvest` the classic fallback |
| `--rms-mix` | 1.0 | 0–1, keep the source's loudness envelope (1) or take the cast's (0) |
| `--device` | cpu:0 | `cpu:0` is deterministic and works everywhere; `cuda:0` if you have it; `mps` is opt-in and known to be unstable |

## Getting models, and the line that matters

- **Train your own** — the recommended path, and the only one where
  provenance is fully yours. Applio exports drop-in model dirs.
- **Community models** — the RVC ecosystem shares thousands of trained
  voices. Treat every one as untrusted input twice over: a `.pth` is a
  pickled artifact (load models only from sources you trust), and a
  voice is a person (use only voices you have the rights and consent
  to use — your own recordings, licensed datasets, synthetic sources).

vox will not help you impersonate anyone. Build voices you own.

## Lineage

The techniques under `vox cast` — VITS, HuBERT/ContentVec, RMVPE,
faiss retrieval, and the RVC software family — are cited per paper in
[REFERENCES.md](REFERENCES.md#vox-cast--rvc-voice-conversion).
