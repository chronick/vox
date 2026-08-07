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

## Status

Early port, in progress. The tools are being moved over from the smpl
repo one at a time and renamed to generic instruments on the way. First
up: `vox ear` (voice-native measurement: F0, formants, harmonics-to-noise,
jitter, shimmer, vibrato).

Planned family, in the organ metaphor the tools already use:

| Tool | Does |
|---|---|
| `vox ear` | voice-native perception: F0, formants F1–F4, HNR, jitter, shimmer, vibrato |
| `vox larynx` | WORLD-vocoder tier: analyze, retune, and re-render pitch/spectrum/aspiration |
| `vox tongue` | a phoneme score: syllables on a beat grid with per-syllable pitch and articulation |
| more | ported as they generalize |

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

Part of the LEMON house: [lemon-agent.dev](https://lemon-agent.dev) ·
[lemon.audio](https://lemon.audio) ·
[smpl](https://chronick.github.io/smpl/).

## License

MIT.
