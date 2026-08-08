# References

vox tools implement published techniques. This file records the papers
and software lineages each tool stands on — house rule: anything we
implement cites its sources. Software links are the lineages actually
shipped; papers are the ideas under them.

## vox cast — RVC voice conversion

Software lineage: [RVC-Project WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
(the architecture), [rvc-python](https://github.com/daswer123/rvc-python)
(the shipped engine), [Applio](https://github.com/IAHispano/Applio)
(the usual training rig; its exports are what `vox cast` loads).

- Kim, Kong, Son — *Conditional Variational Autoencoder with Adversarial
  Learning for End-to-End Text-to-Speech* (VITS), ICML 2021.
  [arXiv:2106.06103](https://arxiv.org/abs/2106.06103) — the synthesis
  backbone RVC adapts.
- Hsu et al. — *HuBERT: Self-Supervised Speech Representation Learning by
  Masked Prediction of Hidden Units*, IEEE/ACM TASLP 2021.
  [arXiv:2106.07447](https://arxiv.org/abs/2106.07447) — the content
  encoder family (`hubert_base`).
- Qian et al. — *ContentVec: An Improved Self-Supervised Speech
  Representation by Disentangling Speakers*, ICML 2022.
  [arXiv:2204.09224](https://arxiv.org/abs/2204.09224) — the
  speaker-disentangled embedder RVC models are trained against.
- Wei et al. — *RMVPE: A Robust Model for Vocal Pitch Estimation in
  Polyphonic Music*, INTERSPEECH 2023.
  [arXiv:2306.15412](https://arxiv.org/abs/2306.15412) — the default
  `--f0-method`.
- Kim, Salamon, Li, Bello — *CREPE: A Convolutional Representation for
  Pitch Estimation*, ICASSP 2018.
  [arXiv:1802.06182](https://arxiv.org/abs/1802.06182) — the `crepe`
  f0 option.
- Morise — *Harvest: A High-Performance Fundamental Frequency Estimator
  from Speech Signals*, INTERSPEECH 2017 — the `harvest` f0 option.
- Johnson, Douze, Jégou — *Billion-Scale Similarity Search with GPUs*,
  IEEE Trans. Big Data 2019.
  [arXiv:1702.08734](https://arxiv.org/abs/1702.08734) — faiss, the
  `.index` retrieval blend behind `--index-rate`.

## vox larynx, vox-core — WORLD analysis/synthesis

- Morise, Yokomori, Ozawa — *WORLD: A Vocoder-Based High-Quality Speech
  Synthesis System for Real-Time Applications*, IEICE Transactions on
  Information and Systems, 2016 — F0/spectral-envelope/aperiodicity
  decomposition behind `analyze`, `render`, `harmonize`, and the
  bass-safe F0 ruler's pyworld leg.
- Morise — *Harvest* (above) — the low-floor F0 estimator in
  `measure_f0_guarded`.
- Software: [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder).

## vox ear — voice measurement

- Jadoul, Thompson, de Boer — *Introducing Parselmouth: A Python
  Interface to Praat*, Journal of Phonetics, 2018 — the shipped bridge.
- Boersma, Weenink — *Praat: Doing Phonetics by Computer* — the
  measurement engine (formants, jitter, shimmer).
- Boersma — *Accurate Short-Term Analysis of the Fundamental Frequency
  and the Harmonics-to-Noise Ratio of a Sampled Sound*, Proceedings of
  the Institute of Phonetic Sciences 17, 1993 — the HNR and
  autocorrelation-F0 methods `ear` reports (and the parselmouth leg of
  the F0 ruler).

## vox corpus — ingest gate

- Baas, van Niekerk, Kamper — *Voice Conversion With Just Nearest
  Neighbors* (kNN-VC), INTERSPEECH 2023.
  [arXiv:2305.18975](https://arxiv.org/abs/2305.18975) — the
  VAD-survivability check exists because kNN-VC's matching-set VAD
  silently drops steady-state material; the gate predicts that fate
  before ingest.

## vox tongue — phoneme score and its render targets

- Liu, Li, Ren, Chen, Zhao — *DiffSinger: Singing Voice Synthesis via
  Shallow Diffusion Mechanism*, AAAI 2022.
  [arXiv:2105.02446](https://arxiv.org/abs/2105.02446) — the
  `emit-ds` export target.
- Radford et al. — *Robust Speech Recognition via Large-Scale Weak
  Supervision* (Whisper), 2022.
  [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) — the
  `[whisper]` extra behind `warp` alignment (and `vox dataset`
  transcript coverage).
- The [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
  (via [`pronouncing`](https://github.com/aparrish/pronouncingpy)) —
  ARPABET syllabification in `compile` and `vox lyric` packets.

## vox bodies, vox syllabank — synthdef voices

- Rodet — *Time-Domain Formant-Wave-Function Synthesis*, Computer Music
  Journal 8(3), 1984 — FOF, the technique `voxFof` approximates with
  SuperCollider Formlet banks.
- Rodet, Potard, Barrière — *The CHANT Project: From the Synthesis of
  the Singing Voice to Synthesis in General*, Computer Music Journal
  8(3), 1984 — the singing-synthesis program FOF came from.
- McCartney — *Rethinking the Computer Music Language: SuperCollider*,
  Computer Music Journal 26(4), 2002 — the NRT render engine under
  every SC body.

## vox carrier — vocoder + verification

- Dudley — *The Vocoder*, Bell Laboratories Record 18, 1939 — the
  channel-vocoder idea (modulator analysis imposed on a carrier) the
  render path descends from.
- Taal, Hendriks, Heusdens, Jensen — *A Short-Time Objective
  Intelligibility Measure for Time-Frequency Weighted Noisy Speech*,
  ICASSP 2010 — STOI.
- Jensen, Taal — *An Algorithm for Predicting the Intelligibility of
  Speech Masked by Modulated Noise Maskers*, IEEE/ACM TASLP 2016 —
  ESTOI, the dev-time intelligibility check (via
  [pystoi](https://github.com/mpariente/pystoi)).

## Datasets referenced in docs

- Wilkins, Seetharaman, Wahl, Pardo — *VocalSet: A Singing Voice
  Dataset*, ISMIR 2018. CC BY 4.0
  ([Zenodo](https://doi.org/10.5281/zenodo.1193957)) — the licensed
  singing corpus the docs recommend as a training base for casts.

`vox vector`'s six axes and `vox flow`'s pattern DSL are original to
vox; their measurements route through the ear/WORLD citations above.
