#!/usr/bin/env bash
# Fresh-install end-to-end: prove every documented command runs as shown, from
# nothing but uv + the public repos. Installs into an ISOLATED tool dir (never
# touches your real `uv tool` installs), then runs the singing guide's six
# commands and the README pipes verbatim. Needs: uv, macOS `say`, network.
set -u
E2E_ROOT="${E2E_ROOT:-$(mktemp -d)}"
export UV_TOOL_DIR="$E2E_ROOT/tools"
export UV_TOOL_BIN_DIR="$E2E_ROOT/bin"
export PATH="$UV_TOOL_BIN_DIR:$PATH"
SMPL_CAS_DIR="$E2E_ROOT/cas"
export SMPL_CAS_DIR
WORK="$E2E_ROOT/work"
mkdir -p "$WORK"
cd "$WORK" || exit 1

declare -a STEPS RESULTS
fail=0
step() {  # name, then the command as remaining args
  local name="$1"; shift
  echo
  echo "=== $name"
  if "$@"; then
    STEPS+=("$name"); RESULTS+=("pass")
  else
    STEPS+=("$name"); RESULTS+=("FAIL")
    fail=1
  fi
}

# ---------------------------------------------------------- installs ----
# smpl core, exactly as the smpl README documents it.
step "install smpl core" uv tool install --quiet \
  "git+https://github.com/chronick/smpl#subdirectory=packages/smpl" \
  --with "git+https://github.com/chronick/smpl#subdirectory=packages/smplstream" \
  --with "git+https://github.com/chronick/smpl#subdirectory=packages/smpl-analysis"

# vox tools, exactly as INSTALL.md documents them (the guide's set + cast).
for t in packages/vox tools/vox-ear tools/vox-larynx tools/vox-vector tools/vox-lyric tools/vox-tongue tools/vox-cast; do
  step "install $t" uv tool install --quiet "git+https://github.com/chronick/vox#subdirectory=$t"
done

# Guard against PATH fallthrough: a failed install must not silently borrow a
# system-wide binary and fake a pass. Both entrypoints must live in the isolated bin.
fresh_bin() {
  local resolved
  resolved="$(command -v "$1" || true)"
  case "$resolved" in
    "$UV_TOOL_BIN_DIR"/*) return 0 ;;
    *) echo "  $1 resolved outside the fresh install: ${resolved:-not found}"; return 1 ;;
  esac
}
step "smpl binary is the fresh install" fresh_bin smpl
step "vox binary is the fresh install" fresh_bin vox

step "vox --help lists tools" sh -c 'vox --help | grep -q "ear"'
step "smpl --help runs" sh -c 'smpl --help > /dev/null'

# ------------------------------------------- the singing guide, verbatim ----
step "guide 1: say speaks" \
  say -v Fred --file-format=WAVE --data-format=LEI16@44100 -o spoken.wav "slow river carry me home"

step "guide 2: lyric gate says keep" sh -c \
  'vox lyric review --delivery sustained --lines "slow river carry me home" --json | grep -q "\"verdict\": \"keep\""'

step "guide 3: compile the score" \
  vox tongue compile --lines "slow river carry me home" --melody "A2,C3,E3,D3,C3" --bpm 90 --score river.yaml

step "guide 4: sing it" sh -c 'vox tongue sing --score river.yaml --out sung.wav > /dev/null'

step "guide 5: stack a choir (smpl|vox pipe)" sh -c \
  'smpl read sung.wav | vox larynx harmonize --chord 0,3,7 --drone | smpl write choir.wav > /dev/null'

step "guide 6: measure it" sh -c \
  'smpl read choir.wav | vox ear describe | vox vector measure | smpl view | grep -q "voice.f0_median_hz"'

# --------------------------------------------------- README pipes ----
step "README: ear report" sh -c \
  'smpl read sung.wav | vox ear describe | smpl view | grep -q "voice.hnr_db"'
step "README: vector diff" sh -c \
  'smpl read sung.wav | vox ear describe | vox vector diff --target "{\"breathiness\":0.2,\"roughness\":0.1}" | grep -q "mean_abs_error"'
step "README: larynx retune" sh -c \
  'smpl read sung.wav | vox larynx render --semitones 2 | smpl write up2.wav > /dev/null'
step "README: lyric review json" sh -c \
  'vox lyric review --delivery percussive --lines "spit the code back|cut the deck to black" --json | grep -q "n_lines"'

# ------------------------------------------------- cast degradation ----
# The documented behavior WITHOUT the 3 GB engine build: status says so,
# and convert fails with the setup hint (never a traceback). Point the
# engine path inside E2E_ROOT so a host machine's real engine can't leak in.
export VOX_RVC_ENGINE="$E2E_ROOT/vox-engine"
step "cast: status reports engine absent" sh -c \
  'vox cast setup --status | grep -q "\"installed\": false"'
mkdir -p "$WORK/fakecast" && : > "$WORK/fakecast/fake.pth"
step "cast: convert degrades with setup hint" sh -c \
  '! (smpl read spoken.wav | vox cast convert --model fakecast >/dev/null 2>cast-err.txt) && grep -q "vox cast setup" cast-err.txt'

# ------------------------------------------------------------ summary ----
echo
echo "== fresh-install E2E summary  (root: $E2E_ROOT)"
for i in "${!STEPS[@]}"; do
  printf '  %-42s %s\n' "${STEPS[$i]}" "${RESULTS[$i]}"
done
exit "$fail"
