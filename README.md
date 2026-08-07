<div align="center">

# vox

**Voice synthesis and voice analysis as pipe tools, on the smpl frame protocol.**

*The voice family of the [smpl](https://github.com/chronick/smpl) toolchain,
grown into its own package: measure a voice like an instrument, control a
render like a score, and verify what came out against what was asked for.*

</div>

---

```bash
smpl read take.wav | vox ear | smpl view
```

`vox` tools read and write the same self-describing NDJSON frames as
`smpl`, so the two toolchains sit in one pipe: smpl handles audio-in,
storage, and reporting; vox contributes the voice-specific stages.

> **Site**: <https://chronick.github.io/vox/> — hear the voices the
> toolchain builds from synthetic sources alone.

## Status

The port from the smpl repo is complete: every tool renamed to a
generic instrument, each with its own tests (`scripts/test-all.sh`
runs all thirteen suites). Live today:

| Tool | Does |
|---|---|
| `vox ear` | voice-native measurement: F0, formants F1–F4, HNR, jitter, shimmer, vibrato |
| `vox larynx` | WORLD-vocoder tier: analyze, retune (formants kept), harmonize into a choir |
| `vox vector` | a six-axis voice coordinate (humanness, breathiness, roughness, intelligibility, multiplicity, spatiality); measure it, and diff it against a target |
| `vox lyric` | lyric prosody verifier + CMUdict packet builder (delivery: sustained or percussive) |
| `vox corpus` | voice-corpus ingest gate: peak-normalize + VAD-survivability |
| `vox flow` | a rhythmic cadence grammar: pattern DSL → grid-placed syllables, say-spat render, named fx chains |
| `vox syllabank` | a syllable-addressable sample bank with a hard provenance contract (source + license or refused) |
| `vox dataset` | the dataset-health doctor: rubric-scored voicebank fitness with shipped model profiles |
| `vox take` | card-driven render-and-self-verify: render, measure on the six axes, report the error |
| `vox bodies` | a registry of named carrier voices with measured fingerprints |
| `vox tongue` | a phoneme score with render paths: say→WORLD singing, bank concat, DiffSinger export, singing warp |
| `vox carrier` | spat words poured into a deep harsh body: vocode + bass chain + dry-diction layer |

Plus `vox-core` (the shared bass-safe F0 ruler and the shipped
SuperCollider synthdefs `voxFof`/`voxGrowl`/`voxSubSaw`/`voxThroat`,
rendered through smpl-synth's NRT bridge).

## Use it with smpl

Every pipe below is real and runs today (install both toolchains first;
see [INSTALL.md](INSTALL.md)):

```bash
# Measure a voice: F0, formants, HNR, vibrato, in an smpl report
smpl read take.wav | vox ear describe | smpl view

# Where does this render sit on the six axes? (ear enriches vector)
smpl read take.wav | vox ear describe | vox vector measure | smpl view

# Did the render land where the target asked? Error becomes a number.
smpl read take.wav | vox ear describe | vox vector diff --target '{"breathiness":0.2,"roughness":0.1}'

# Retune a voice two semitones up without chipmunking, back into the store
smpl read take.wav | vox larynx render --semitones 2 | smpl write up2.wav

# Stack one voice into a chord-locked choir with a drone under it
smpl read take.wav | vox larynx harmonize --chord 0,3,7 --drone | smpl write choir.wav

# Verify lyrics before any audio exists
vox lyric review --delivery percussive --lines "spit the code back|cut the deck to black" --json
```

## Design

- **Frames, not files.** Every stage passes smpl-protocol frames;
  heavy audio stays content-addressed on disk. The protocol contract
  lives in the smpl repo (`spec.md`) and vox conforms to it.
- **Two tiers, like smpl.** A light `vox` dispatcher; each heavy tool
  in its own isolated install, found by name (`vox ear` hands off to
  `vox-ear`).
- **Generic instruments.** Tool names, measurements, and controls are
  standard voice-science vocabulary (formants, HNR, vibrato), not any
  one project's language.
- **Install only what you use.** Each tool is its own isolated install
  with its own dependency story; [INSTALL.md](INSTALL.md) has the
  dependency matrix, the Python-pin story, and the optional pieces.

Part of the LEMON house: [lemon-agent.dev](https://lemon-agent.dev) ·
[lemon.audio](https://lemon.audio) ·
[smpl](https://chronick.github.io/smpl/).

## License

MIT.
