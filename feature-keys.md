# vox feature-key registry

The canonical key names, units, and shapes for `feature` frames emitted by
vox tools, in the image of the smpl registry (smpl's `feature-keys.md`).
Every vox tool that emits a `feature` frame MUST use a key registered here;
adding a measurement means adding a row here first, then emitting it.

Conventions follow the smpl spec: domain keys carry a short prefix and,
where a bare number would be ambiguous, a unit suffix (`_hz`, `_db`,
`_cents`).

## Registry

| Key | Unit | Stat | Emitted by |
|---|---|---|---|
| `voice.voiced_frac` | 0–1 | scalar | ear |
| `voice.f0_median_hz` | Hz | scalar (None if unvoiced) | ear |
| `voice.f0_range_semitones` | semitones | scalar (None if unvoiced) | ear |
| `voice.f1_hz` … `voice.f4_hz` | Hz | scalar per formant (None if unvoiced) | ear |
| `voice.hnr_db` | dB (harmonics-to-noise; breathiness inverse) | scalar (None if unvoiced) | ear |
| `voice.jitter_local` | ratio (0–1, pitch perturbation) | scalar (None if unvoiced) | ear |
| `voice.shimmer_local_db` | dB (amplitude perturbation) | scalar (None if unvoiced) | ear |
| `voice.vibrato_rate_hz` | Hz (3–8 Hz band) | scalar (None if flat/short) | ear |
| `voice.vibrato_extent_cents` | cents (2×RMS of detrended F0) | scalar (None if flat/short) | ear |
| `voice.spectral_flatness_hf` | 0–1 (HF geo/arith power ratio; breathiness support) | scalar | ear |
| `voice.inharmonicity` | 0–1 (off-harmonic-grid energy; clangor support) | scalar | ear |
