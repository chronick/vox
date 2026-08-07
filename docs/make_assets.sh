#!/usr/bin/env bash
# Build the site's voices through the REAL vox pipelines — the examples double as an
# end-to-end integration proof. Deterministic given this machine's `say` voice + SC.
# Needs: the tool venvs (uv sync in tools/*), say, ffmpeg, SuperCollider.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="$HERE/assets"
WORK="$(mktemp -d)"
SMPL_CAS_DIR="$(mktemp -d)/cas"
export SMPL_CAS_DIR
mkdir -p "$OUT"

TONGUE_PY="$REPO/tools/vox-tongue/.venv/bin/python"
CARRIER_PY="$REPO/tools/vox-carrier/.venv/bin/python"
TAKE_PY="$REPO/tools/vox-take/.venv/bin/python"
BODIES="$REPO/tools/vox-bodies/.venv/bin/vox-bodies"

# ------------------------------------------------------- the choir voice ----
echo "-- building the choir (lyric -> score -> say -> WORLD -> harmonize)"
"$TONGUE_PY" - "$WORK" <<'PYEOF'
import sys

import numpy as np
import soundfile as sf
from vox_larynx import world
from vox_lyric.packet import build_packet
from vox_tongue import render as render_mod
from vox_tongue.compile import compile_packet

work = sys.argv[1]
SR, BPM, BREATH = 44100, 90.0, 0.35
MELODY = ["A2", "C3", "E3", "D3", "C3"]
LINES = ["Slow river carry me home", "Open water hold the line"]

chunks = []
gap = np.zeros(int(BREATH * SR))
for li, line in enumerate(LINES):
    mel = MELODY[li % len(MELODY):] + MELODY[:li % len(MELODY)]
    packet = build_packet([line], "sustained")
    score = compile_packet(packet, mel, bpm=BPM)
    samples, _manifest = render_mod.render(score, sr=SR, voice="Fred")
    chunks.append(samples.astype("float64"))
    chunks.append(gap.copy())
chunks.pop()
dry = np.concatenate(chunks)
sf.write(f"{work}/choir-dry.wav", (dry / max(abs(dry).max(), 1e-9) * 0.85).astype("float32"), SR)

mix, params = world.harmonize(dry, SR, chord=(0, 3, 7), mode="follow", drone=True)
mix = mix / max(abs(mix).max(), 1e-9) * 0.85
sf.write(f"{work}/choir.wav", mix.astype("float32"), SR)
print("choir voices:", params["voices"], "chord:", params["chord"])
PYEOF

# ------------------------------------------------ the deep carrier voice ----
echo "-- building the deep carrier (lyric -> flow spit -> growl body -> vocode -> bass chain)"
CARRIER_JSON="$("$CARRIER_PY" - "$WORK" <<'PYEOF'
import json
import sys

import soundfile as sf
from vox_carrier.carrier import render_carrier_verse
from vox_core import measure_f0_guarded

work = sys.argv[1]
LINES = ["Kick the pattern back to the top", "Cut the deck and count to ten"]
r = render_carrier_verse(LINES, 142, "growl-55", sr=44100, ess_mix=0.30, dry_db=-14.0)
sf.write(f"{work}/carrier-dry.wav", r["modulator"], r["modulator_sr"], subtype="FLOAT")
sf.write(f"{work}/carrier.wav", r["final"], r["sr"], subtype="FLOAT")
body_f0 = measure_f0_guarded(r["body"], r["sr"], target_hint=55.0)
print(json.dumps({"params": r["params"], "body_f0": body_f0}))
PYEOF
)"

# --------------------------------------------------- the bodies gallery ----
echo "-- rendering the bodies gallery (SuperCollider)"
for b in growl-55 subsaw-55 throat-60 fof-a-180 fof-impossible; do
  "$BODIES" render "$b" --out "$WORK/body-$b.wav" --dur 2.0 >/dev/null
done

# ------------------------------------------------------- measurements ----
echo "-- measuring the built voices (ear -> vector)"
MEASURES="$("$TAKE_PY" - "$WORK" <<'PYEOF'
import json
import sys

import numpy as np
import soundfile as sf
from vox_ear import descriptors
from vox_vector import axes

work = sys.argv[1]
out = {}
for name in ("choir", "carrier"):
    data, sr = sf.read(f"{work}/{name}.wav", dtype="float64", always_2d=True)
    x = np.ascontiguousarray(data.mean(axis=1))
    voice = descriptors.describe(x, sr)
    vec = axes.measure(x, sr, upstream=voice)
    out[name] = {"vector": vec["vector"],
                 "hnr_db": voice.get("voice.hnr_db"),
                 "f0_median_hz": voice.get("voice.f0_median_hz"),
                 "vibrato_rate_hz": voice.get("voice.vibrato_rate_hz")}
print(json.dumps(out))
PYEOF
)"

# ------------------------------------------- the say-to-singing guide ----
echo "-- building the guide steps (spoken -> score -> sung -> choir)"
GUIDE_LINE="slow river carry me home"
say -v Fred --file-format=WAVE --data-format=LEI16@44100 -o "$WORK/guide-spoken.wav" "$GUIDE_LINE"

GUIDE_JSON="$("$TONGUE_PY" - "$WORK" "$GUIDE_LINE" <<'PYEOF'
import json
import sys

import numpy as np
import soundfile as sf
from vox_larynx import world
from vox_tongue import render as render_mod
from vox_tongue import schema
from vox_tongue.compile import compile as compile_score

work, line = sys.argv[1], sys.argv[2]
SR = 44100
score = compile_score([line], ["A2", "C3", "E3", "D3", "C3"], bpm=90)
open(f"{work}/guide-score.yaml", "w").write(schema.to_yaml(score))

sung, manifest = render_mod.render(score, sr=SR, voice="Fred")
sung64 = sung.astype("float64")
sf.write(f"{work}/guide-sung.wav",
         (sung64 / max(abs(sung64).max(), 1e-9) * 0.85).astype("float32"), SR)

mix, params = world.harmonize(sung64, SR, chord=(0, 3, 7), mode="follow", drone=True)
mix = mix / max(abs(mix).max(), 1e-9) * 0.85
sf.write(f"{work}/guide-choir.wav", mix.astype("float32"), SR)

print(json.dumps({"n_syllables": len(score["syllables"]),
                  "notes": [s["note"] for s in score["syllables"]],
                  "voices": params["voices"], "chord": params["chord"]}))
PYEOF
)"

GUIDE_MEASURES="$("$TAKE_PY" - "$WORK" <<'PYEOF'
import json
import sys

import numpy as np
import soundfile as sf
from vox_ear import descriptors
from vox_vector import axes

work = sys.argv[1]
out = {}
for name in ("guide-sung", "guide-choir"):
    data, sr = sf.read(f"{work}/{name}.wav", dtype="float64", always_2d=True)
    x = np.ascontiguousarray(data.mean(axis=1))
    voice = descriptors.describe(x, sr)
    vec = axes.measure(x, sr, upstream=voice)
    out[name] = {"f0_median_hz": voice.get("voice.f0_median_hz"),
                 "spatiality": vec["vector"]["spatiality"],
                 "hnr_db": voice.get("voice.hnr_db")}
print(json.dumps(out))
PYEOF
)"
head -28 "$WORK/guide-score.yaml" > "$OUT/guide-score.txt"

# ------------------------------------------------- waveform SVG renders ----
# Page graphics of the real audio files (cool grammar: cyan = dry/in, violet = built voice).
wf() { # in.wav out.svg color
  "$TAKE_PY" - "$1" "$2" "$3" <<'PYEOF'
import sys

import numpy as np
import soundfile as sf

wav, out, color = sys.argv[1], sys.argv[2], sys.argv[3]
data, sr = sf.read(wav, always_2d=True)
data = data.mean(axis=1)
W, H = 720, 150
seg = max(1, len(data) // W)
mid, amp = H / 2, H / 2 - 6
tops, bots = [], []
for i in range(W):
    c = data[i * seg:(i + 1) * seg]
    if len(c) == 0:
        c = np.zeros(1)
    tops.append(f"{i},{mid - max(0.006, float(c.max())) * amp:.1f}")
    bots.append(f"{i},{mid - min(-0.006, float(c.min())) * amp:.1f}")
path = "M" + " L".join(tops + bots[::-1]) + " Z"
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
       f'preserveAspectRatio="none" role="img">'
       f'<line x1="0" y1="{mid}" x2="{W}" y2="{mid}" stroke="{color}" stroke-width="0.5" opacity="0.35"/>'
       f'<path d="{path}" fill="{color}" fill-opacity="0.55" stroke="{color}" stroke-width="1"/></svg>')
open(out, "w").write(svg)
PYEOF
}
CYAN="#7fd6e8"
VIOLET="#b9a5f2"
wf "$WORK/choir-dry.wav"   "$OUT/wf-choir-dry.svg"   "$CYAN"
wf "$WORK/choir.wav"       "$OUT/wf-choir.svg"       "$VIOLET"
wf "$WORK/carrier-dry.wav" "$OUT/wf-carrier-dry.svg" "$CYAN"
wf "$WORK/carrier.wav"     "$OUT/wf-carrier.svg"     "$VIOLET"
for b in growl-55 subsaw-55 throat-60 fof-a-180 fof-impossible; do
  wf "$WORK/body-$b.wav" "$OUT/wf-$b.svg" "$VIOLET"
done
wf "$WORK/guide-spoken.wav" "$OUT/wf-guide-spoken.svg" "$CYAN"
wf "$WORK/guide-sung.wav"   "$OUT/wf-guide-sung.svg"   "$VIOLET"
wf "$WORK/guide-choir.wav"  "$OUT/wf-guide-choir.svg"  "$VIOLET"

# ------------------------------------------------------------ package ----
for f in choir-dry choir carrier-dry carrier body-growl-55 body-subsaw-55 body-throat-60 \
         body-fof-a-180 body-fof-impossible guide-spoken guide-sung guide-choir; do
  cp "$WORK/$f.wav" "$OUT/$f.wav"
done

"$TAKE_PY" - "$OUT/numbers.json" "$CARRIER_JSON" "$MEASURES" "$GUIDE_JSON" "$GUIDE_MEASURES" <<'PYEOF'
import json
import sys

receipt = {"carrier": json.loads(sys.argv[2]), "measures": json.loads(sys.argv[3]),
           "guide": json.loads(sys.argv[4]), "guide_measures": json.loads(sys.argv[5])}
json.dump(receipt, open(sys.argv[1], "w"), indent=1)
print(json.dumps(receipt, indent=1))
PYEOF

rm -rf "$WORK"
echo "done → $OUT"
