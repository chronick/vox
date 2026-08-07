// voxFof — FOF/Chant-style formant singing.
//
// An abstract voice-box with zero extra deps: a glottal impulse train at F0
// excites a bank of five Formlet resonators (SuperCollider's built-in FOF-like filter — a
// resonator with independent attack/decay, i.e. formant centre + bandwidth), summed with
// per-formant amplitudes. Vowel identity is the formant TABLE, passed as args, so the same
// SynthDef sings /a/, /i/, /u/ … AND impossible-geometry stacks (formants above any human
// vocal tract) — the otherworldliness dial. Vibrato rides F0. Rendered offline through
// smpl-synth's NRT bridge (sclang → scsynth NRT).
//
// Defaults are /a/ (as in "father"): F1..F5 = 730/1090/2440/3400/4500 Hz.
SynthDef(\voxFof, { |out=0, freq=180, amp=0.3, dur=1.0, vibRate=5, vibDepth=0.02,
                    f1=730, f2=1090, f3=2440, f4=3400, f5=4500,
                    a1=1.0, a2=0.5, a3=0.25, a4=0.12, a5=0.05|
    var vib, f0, src, formants, sig, env;
    // Vibrato as a small multiplicative wobble on F0 (±vibDepth around 1.0).
    vib = SinOsc.kr(vibRate, 0, vibDepth, 1.0);
    f0 = freq * vib;
    // Glottal source: a band-limited impulse train (rich harmonics to be shaped by the formants).
    src = Blip.ar(f0, 40);
    // Five FOF resonators. Formlet.ar(in, freq, attacktime, decaytime); shorter decay = wider
    // bandwidth. Decays taper for higher formants (narrower low formants, broader highs).
    formants = Formlet.ar(src, [f1, f2, f3, f4, f5], 0.005,
                          [0.045, 0.035, 0.025, 0.02, 0.015]) * [a1, a2, a3, a4, a5];
    sig = Mix(formants);
    // linen over the note so the NRT render has clean fade-in/out (doneAction frees the synth).
    env = EnvGen.kr(Env.linen(0.05, (dur - 0.1).max(0.05), 0.05), doneAction: 2);
    Out.ar(out, Pan2.ar(sig * env * amp, 0));
}).add;
