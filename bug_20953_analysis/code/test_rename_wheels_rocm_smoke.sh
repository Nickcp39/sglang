#!/usr/bin/env bash
# Smoke test for ROCm rename script (unpack/pack + METADATA), without /opt/rocm-*.
# Uses SGL_KERNEL_ROCM_SUFFIX_OVERRIDE (e.g. +rocm702) to match upstream +rocm${ver_abrv} style.
# Run from repo root: bash bug_20953_analysis/code/test_rename_wheels_rocm_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROCM_SCRIPT="${ROOT}/3rdparty/amd/wheel/sgl-kernel/rename_wheels_rocm.sh"
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

PY=""
try_py() {
  local c="$1"
  [[ -z "$c" ]] && return 1
  "$c" -c "import wheel" 2>/dev/null || return 1
  PY="$c"
  return 0
}
if [[ -n "${PYTHON_ROOT_PATH:-}" ]]; then
  try_py "${PYTHON_ROOT_PATH}/bin/python" || try_py "${PYTHON_ROOT_PATH}/python" || true
fi
command -v python >/dev/null 2>&1 && try_py "$(command -v python)" || true
command -v python3 >/dev/null 2>&1 && try_py "$(command -v python3)" || true
if [[ -z "$PY" ]]; then
  echo "Need python with wheel" >&2
  exit 1
fi

[[ -f "$ROCM_SCRIPT" ]] || { echo "Missing $ROCM_SCRIPT" >&2; exit 1; }

STAGE="${WORKDIR}/sgl_kernel-0.4.0"
mkdir -p "${STAGE}/sgl_kernel" "${STAGE}/sgl_kernel-0.4.0.dist-info"
echo 'x = 1' > "${STAGE}/sgl_kernel/__init__.py"
cat > "${STAGE}/sgl_kernel-0.4.0.dist-info/METADATA" << 'EOF'
Metadata-Version: 2.1
Name: sgl_kernel
Version: 0.4.0
EOF
cat > "${STAGE}/sgl_kernel-0.4.0.dist-info/WHEEL" << 'EOF'
Wheel-Version: 1.0
Generator: test
Root-Is-Purelib: false
Tag: cp310-abi3-linux_x86_64
EOF

DIST="${WORKDIR}/dist"
mkdir -p "$DIST"
(cd "$WORKDIR" && "$PY" -m wheel pack "${STAGE##*/}" --dest-dir "$DIST")

WHL=$(echo "$DIST"/*.whl)
[[ -f "$WHL" ]] || { echo "wheel pack failed" >&2; exit 1; }

cp -a "$ROCM_SCRIPT" "${WORKDIR}/rename_wheels_rocm.sh"
chmod +x "${WORKDIR}/rename_wheels_rocm.sh"

export SGL_KERNEL_ROCM_SUFFIX_OVERRIDE="${SGL_KERNEL_ROCM_SUFFIX_OVERRIDE:-+rocm702}"
mkdir -p "${WORKDIR}/dist"
mv "$WHL" "${WORKDIR}/dist/"
(cd "$WORKDIR" && ./rename_wheels_rocm.sh)

OUT=$(echo "${WORKDIR}/dist"/*.whl)
[[ -f "$OUT" ]] || { echo "expected output wheel" >&2; exit 1; }

echo "Output: $OUT"
echo "Using SGL_KERNEL_ROCM_SUFFIX_OVERRIDE=${SGL_KERNEL_ROCM_SUFFIX_OVERRIDE}"

unzip -p "$OUT" '*/METADATA' | grep -qF "Version: 0.4.0${SGL_KERNEL_ROCM_SUFFIX_OVERRIDE}" || {
  echo "METADATA Version mismatch"
  unzip -p "$OUT" '*/METADATA'
  exit 1
}

unzip -p "$OUT" '*/WHEEL' | grep -q 'manylinux2014_x86_64' || {
  echo "WHEEL Tag should contain manylinux2014_x86_64"
  unzip -p "$OUT" '*/WHEEL'
  exit 1
}

bash "${ROOT}/bug_20953_analysis/code/verify_wheel_metadata.sh" "$OUT"

echo "OK: rename_wheels_rocm smoke test passed."
