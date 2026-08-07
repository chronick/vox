// voxGrowl — deep FM growl (deep carrier family).
//
// A single FM pair: a ~55 Hz sine CARRIER frequency-modulated by a modulator at a deliberately
// NON-integer ratio (~2.37) so the sidebands land off the harmonic grid — inharmonic, animal,
// wrong. A high modulation index makes it harsh; a slow LFO on the index breathes the growl open
// and shut. tanh saturation adds odd harmonics (the snarl); a gentle LPF ~2.5 kHz tames the fizz
// so it reads as a body, not a buzz. Rendered offline through smpl-synth's NRT bridge.
SynthDef(\voxGrowl, { |out=0, freq=55, amp=0.3, dur=2.0, ratio=2.37, index=8,
                      idxLfoRate=0.7, idxLfoDepth=4, lpf=2500, drive=2|
    var idx, modFreq, mod, car, sig, env;
    // Modulation index breathes: base index ± an LFO swing (slow, so the growl opens over the note).
    idx = (index + SinOsc.kr(idxLfoRate, 0, idxLfoDepth)).max(0);
    modFreq = freq * ratio;                                   // non-integer ratio → inharmonic sidebands
    // FM: frequency deviation = index * modulator-frequency (classic Chowning index scaling).
    mod = SinOsc.ar(modFreq) * (modFreq * idx);
    car = SinOsc.ar(freq + mod);
    sig = (car * drive).tanh;                                 // saturation — odd harmonics, the snarl
    sig = LPF.ar(sig, lpf);                                   // gentle low-pass — a body, not a buzz
    env = EnvGen.kr(Env.linen(0.05, (dur - 0.1).max(0.05), 0.05), doneAction: 2);
    Out.ar(out, Pan2.ar(sig * env * amp, 0));
}).add;
