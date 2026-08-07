# Installing vox

vox is a compilation of small packages, not one monolith. Each tool
installs into its own isolated environment and is found by name, so you
install only what you use and a heavy dependency in one tool can never
break another. This page is the map.

Everything installs with [uv](https://docs.astral.sh/uv/).

## The shape

- **`vox`** (the dispatcher) is a tiny dependency-free command:
  `vox ear …` runs the separately installed `vox-ear`. If a tool is
  missing, the error prints the exact install command for it.
- **Tools** live under `tools/` and install independently. Most speak
  the [smpl](https://github.com/chronick/smpl) frame protocol on
  stdin/stdout, so they sit inside smpl pipes.
- **`vox-core`** is a shared library some tools depend on (pulled in
  automatically; you never install it directly).

## Install

```bash
# the dispatcher
uv tool install git+https://github.com/chronick/vox#subdirectory=packages/vox

# then any tools you want (each is standalone):
uv tool install git+https://github.com/chronick/vox#subdirectory=tools/vox-ear
uv tool install git+https://github.com/chronick/vox#subdirectory=tools/vox-larynx
uv tool install git+https://github.com/chronick/vox#subdirectory=tools/vox-vector
uv tool install git+https://github.com/chronick/vox#subdirectory=tools/vox-lyric
uv tool install git+https://github.com/chronick/vox#subdirectory=tools/vox-corpus
uv tool install git+https://github.com/chronick/vox#subdirectory=tools/vox-flow
uv tool install git+https://github.com/chronick/vox#subdirectory=tools/vox-syllabank
uv tool install git+https://github.com/chronick/vox#subdirectory=tools/vox-dataset
uv tool install git+https://github.com/chronick/vox#subdirectory=tools/vox-take
uv tool install git+https://github.com/chronick/vox#subdirectory=tools/vox-bodies
uv tool install git+https://github.com/chronick/vox#subdirectory=tools/vox-tongue
```

For the pipe examples you will also want the smpl core (`smpl read`,
`smpl view`); its install command is in the
[smpl README](https://github.com/chronick/smpl#install).

## Dependency matrix

| Tool | Python | Heavy deps | Optional | System binaries | Without the optional pieces |
|---|---|---|---|---|---|
| `vox` (dispatcher) | ≥3.11 | none | — | — | — |
| `vox-ear` | 3.10–3.12 | praat-parselmouth | — | — | — |
| `vox-larynx` | 3.10–3.12 | pyworld | — | — | — |
| `vox-vector` | ≥3.10 | none (numpy/scipy) | — | — | upstream `vox ear` frames enrich it; without them it falls back to its own signal proxies |
| `vox-lyric` | ≥3.10 | none (pure text) | pronouncing (CMUdict) | — | phones are `null`, syllable counts fall back to a heuristic |
| `vox-corpus` | ≥3.10 | none (numpy/soundfile) | — | — | — |
| `vox-flow` | ≥3.10 | none (numpy/soundfile) | — | `say`, `ffmpeg` | score compiles anywhere; render needs `say`, chains need `ffmpeg` |
| `vox-syllabank` | 3.10–3.12 | pyworld | `[fof]` extra (smpl-synth) | `say`; SuperCollider for FOF | seeds from `say` alone; FOF vowels skip without SC |
| `vox-dataset` | 3.10–3.12 | pyworld | `[whisper]` extra | — | phone coverage reads "unknown" without transcripts/whisper |
| `vox-take` | 3.10–3.12 | via siblings (pyworld, parselmouth) | — | `say`/`ffmpeg` for flow cards | larynx cards run without any system binary |
| `vox-bodies` | 3.10–3.12 | via siblings + smpl-synth | — | SuperCollider for SC engines | larynx-recipe bodies render without SC |
| `vox-tongue` | 3.10–3.12 | pyworld (+ siblings) | `[whisper]` extra | `say` for the sing path | compile/emit-ds work anywhere; sing needs `say`; warp needs whisper |
| `vox-core` (library) | 3.10–3.12 | pyworld | praat-parselmouth | — | F0 ruler degrades to pyworld-only; also ships the voxFof/voxGrowl/voxSubSaw/voxThroat synthdefs as package data |

Notes on the pins:

- **Why 3.10–3.12 on some tools**: `pyworld` and `praat-parselmouth`
  are compiled wheels with no CPython 3.13+ builds yet. Isolated
  installs mean this pin never constrains the rest of your system.
- `pyworld` also needs `setuptools<81` (it imports `pkg_resources`);
  the affected tools pin this for you.
- smpl interop needs `smplstream`, which every frame-speaking tool
  declares as a git dependency on the smpl repo; uv resolves it
  automatically.

## Check what you have

```bash
vox --help        # lists known tools, with the install command for each
vox ear --help    # errors with the install command if vox-ear is absent
```

## Development

```bash
git clone https://github.com/chronick/vox && cd vox
# per tool:
cd tools/vox-ear && uv sync && uv run pytest -q
# the workspace packages (dispatcher + vox-core):
cd ../.. && uv sync && uv run pytest packages -q
```
