#!/usr/bin/env bash
# Run every package's and tool's test suite in its own environment; print a summary table.
# The one-command correctness gate for the whole repo. Exits nonzero if anything fails.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

declare -a NAMES RESULTS
fail=0

run_suite() {
  local dir="$1"
  local name
  name="$(basename "$dir")"
  if [ ! -d "$dir/tests" ]; then
    return
  fi
  echo "=== $name"
  if (cd "$dir" && uv run --quiet pytest -q); then
    NAMES+=("$name"); RESULTS+=("pass")
  else
    NAMES+=("$name"); RESULTS+=("FAIL")
    fail=1
  fi
}

for d in "$HERE"/packages/*/ "$HERE"/tools/*/; do
  run_suite "${d%/}"
done

echo
echo "== summary"
for i in "${!NAMES[@]}"; do
  printf '  %-16s %s\n' "${NAMES[$i]}" "${RESULTS[$i]}"
done
exit "$fail"
