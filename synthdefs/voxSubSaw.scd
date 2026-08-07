// voxSubSaw — massive detuned-saw sub (deep carrier family).
//
// Three sawtooths around ~55 Hz, detuned ±12 cents so they beat slowly into a wide, synthetic
// wall. tanh drive fattens them; a RESONANT low-pass sweeps 600 → 180 Hz over the note (the classic
// "vwoom" filter fall) so the sound arrives bright and settles into a growl. A sine an octave
// BELOW the fundamental, mixed in low, plants the sub weight. Massive, synthetic — the antithesis
// of the FM growl's animal harshness. Rendered offline through smpl-synth's NRT bridge.
SynthDef(\voxSubSaw, { |out=0, freq=55, amp=0.3, dur=2.0, detune=12, drive=3,
                       lpfStart=600, lpfEnd=180, rq=0.3, subAmp=0.5|
    var freqs, saws, sub, sig, env, cutoff;
    // Three voices at 0 / ±detune cents (cents → ratio via 2**(cents/1200)).
    freqs = freq * (2 ** ([detune.neg, 0, detune] / 1200));
    saws = Mix(Saw.ar(freqs, [0.9, 1.0, 0.9]));
    saws = (saws * drive).tanh;                               // drive → fatten, add harmonics
    cutoff = XLine.kr(lpfStart, lpfEnd, dur);                 // resonant filter FALL over the note
    saws = RLPF.ar(saws, cutoff, rq);
    sub = SinOsc.ar(freq * 0.5) * subAmp;                     // pure sine an octave below — sub weight
    sig = saws + sub;
    env = EnvGen.kr(Env.linen(0.05, (dur - 0.1).max(0.05), 0.05), doneAction: 2);
    Out.ar(out, Pan2.ar(sig * env * amp, 0));
}).add;
