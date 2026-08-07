// voxThroat — a giant's vocal tract (deep carrier family).
//
// An impulse train at ~60 Hz (the glottal source of a huge throat) driven through THREE low
// Formlet formants — 300 / 700 / 1200 Hz — the resonances of a vocal tract far larger than any
// human's, so the vowel identity is legible but the scale is wrong. A little white-noise breath is
// mixed in PRE-formant, so the breath is coloured by the same tract (aspirated, alive). Vocal, but
// wrong — the uncanny giant. Rendered offline through smpl-synth's NRT bridge.
SynthDef(\voxThroat, { |out=0, freq=60, amp=0.3, dur=2.0,
                       f1=300, f2=700, f3=1200, breath=0.05,
                       a1=1.0, a2=0.6, a3=0.3|
    var src, formants, sig, env;
    // Seed the noise so renders are deterministic: measured fingerprints and caching both need
    // the same render for the same args, and unseeded WhiteNoise drifts run-to-run.
    RandSeed.ir(1, 20260711);
    // Glottal source: impulse train (rich harmonics) + a touch of breath, both shaped by the tract.
    src = Impulse.ar(freq) + WhiteNoise.ar(breath);
    // Three formant resonators (shorter decay = wider bandwidth; low formants ring longer).
    formants = Formlet.ar(src, [f1, f2, f3], 0.005, [0.04, 0.03, 0.02]) * [a1, a2, a3];
    sig = Mix(formants);
    env = EnvGen.kr(Env.linen(0.05, (dur - 0.1).max(0.05), 0.05), doneAction: 2);
    Out.ar(out, Pan2.ar(sig * env * amp, 0));
}).add;
