#!/usr/bin/env bash
# Quick check: METADATA Version / WHEEL Tag vs wheel filename (Issue #20953 layer A).
# Usage: bash bug_20953_analysis/code/verify_wheel_metadata.sh path/to/some.whl
set -euo pipefail

WHL="${1:-}"
if [[ -z "$WHL" || ! -f "$WHL" ]]; then
  echo "Usage: $0 <path-to.whl>" >&2
  exit 2
fi

echo "=== File: $WHL ==="
BASE="$(basename "$WHL")"

echo "--- METADATA (Version) ---"
META_V=$(unzip -p "$WHL" '*/METADATA' 2>/dev/null | grep -m1 '^Version:' || true)
echo "${META_V:-<missing>}"

echo "--- WHEEL (Tag lines) ---"
unzip -p "$WHL" '*/WHEEL' 2>/dev/null | grep -E '^(Tag|Root-Is-Purelib):' || echo "<missing>"

echo "--- Filename vs Version (heuristic) ---"
if [[ "$BASE" == *"+cu"* ]]; then
  if [[ "${META_V:-}" != *"+cu"* ]]; then
    echo "FAIL: filename suggests +cu local version but METADATA Version has no +cu (pip may reject index install)."
    exit 1
  fi
  echo "OK: both filename and METADATA mention +cu"
elif [[ "$BASE" == *"+rocm"* ]]; then
  if [[ "${META_V:-}" != *"+rocm"* ]]; then
    echo "FAIL: filename suggests +rocm but METADATA Version has no +rocm."
    exit 1
  fi
  echo "OK: both filename and METADATA mention +rocm"
else
  echo "INFO: no +cu/+rocm in filename; METADATA should match base version only."
fi

echo "Done."
