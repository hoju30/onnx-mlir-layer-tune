#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"
if [[ -f "${script_dir}/px1_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${script_dir}/px1_env.sh"
elif [[ -f "${src_root}/px1_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${src_root}/px1_env.sh"
else
  export SOFTPOSIT_ROOT="${SOFTPOSIT_ROOT:-/home/lai/mlir_toy/SoftPosit/SoftPosit}"
  export SOFTPOSIT_PX1_SRC="${SOFTPOSIT_PX1_SRC:-${SOFTPOSIT_ROOT}/source}"
  export SOFTPOSIT_PX1_INC="${SOFTPOSIT_PX1_INC:-${SOFTPOSIT_PX1_SRC}/include}"
  export SOFTPOSIT_PX1_ARCH_INC="${SOFTPOSIT_PX1_ARCH_INC:-${SOFTPOSIT_PX1_SRC}/8086-SSE}"
  export SOFTPOSIT_PX1_PLATFORM_INC="${SOFTPOSIT_PX1_PLATFORM_INC:-${SOFTPOSIT_ROOT}/build/Linux-x86_64-GCC}"
  export SOFTPOSIT_PX1_LIB="${SOFTPOSIT_PX1_LIB:-${src_root}/.deps/softposit-px1/libsoftposit_full.so}"
  export SOFTPOSIT_PX1_LIB_DIR="${SOFTPOSIT_PX1_LIB_DIR:-$(dirname "${SOFTPOSIT_PX1_LIB}")}"
  export POSIT_PX1_SOFTPOSIT_BUILD_CFLAGS="${POSIT_PX1_SOFTPOSIT_BUILD_CFLAGS:--DSOFTPOSIT_FAST_INT64 -I${SOFTPOSIT_PX1_INC} -I${SOFTPOSIT_PX1_ARCH_INC} -I${SOFTPOSIT_PX1_PLATFORM_INC}}"
fi

force_rebuild=0
if [[ "${1:-}" == "--rebuild" ]]; then
  force_rebuild=1
fi

mkdir -p "${SOFTPOSIT_PX1_LIB_DIR}"

if [[ ! -f "${SOFTPOSIT_PX1_SRC}/pX1_add.c" ]]; then
  echo "ERROR: ${SOFTPOSIT_PX1_SRC}/pX1_add.c not found"
  echo "SOFTPOSIT_PX1_SRC appears to be a non-PX1 SoftPosit source tree."
  echo "Current settings:"
  echo "  SOFTPOSIT_ROOT=${SOFTPOSIT_ROOT}"
  echo "  SOFTPOSIT_PX1_SRC=${SOFTPOSIT_PX1_SRC}"
  echo "Please point SOFTPOSIT_ROOT/SOFTPOSIT_PX1_SRC to a SoftPosit tree that contains pX1_*.c."
  exit 2
fi

if [[ ${force_rebuild} -eq 0 && -f "${SOFTPOSIT_PX1_LIB}" ]]; then
  if nm -D "${SOFTPOSIT_PX1_LIB}" | grep -Eq '[[:space:]]pX1_add(@@.*)?$'; then
    if ! nm -D "${SOFTPOSIT_PX1_LIB}" | grep -Eq '[[:space:]]qX1_fdp_add(@@.*)?$'; then
      echo "WARN: ${SOFTPOSIT_PX1_LIB} does not export qX1_fdp_add/qX1_to_pX1."
      echo "      Current runtime will use p8e1 quire-compatible high-precision fallback."
    fi
    echo "Reusing ${SOFTPOSIT_PX1_LIB}"
    exit 0
  fi
fi

gcc -O3 -fPIC -shared \
  "${SOFTPOSIT_PX1_SRC}"/*.c \
  ${POSIT_PX1_SOFTPOSIT_BUILD_CFLAGS} \
  -o "${SOFTPOSIT_PX1_LIB}"

if ! nm -D "${SOFTPOSIT_PX1_LIB}" | grep -Eq '[[:space:]]pX1_add(@@.*)?$'; then
  echo "ERROR: ${SOFTPOSIT_PX1_LIB} does not export pX1_add"
  echo "Build inputs:"
  echo "  SOFTPOSIT_PX1_SRC=${SOFTPOSIT_PX1_SRC}"
  echo "  SOFTPOSIT_PX1_INC=${SOFTPOSIT_PX1_INC}"
  echo "Exported SoftPosit symbols (sample):"
  nm -D "${SOFTPOSIT_PX1_LIB}" | grep -E ' pX[0-9]_|\bp8_|\bp16_|\bp32_' | head -n 20 || true
  exit 2
fi

if ! nm -D "${SOFTPOSIT_PX1_LIB}" | grep -Eq '[[:space:]]qX1_fdp_add(@@.*)?$'; then
  echo "WARN: ${SOFTPOSIT_PX1_LIB} does not export qX1_fdp_add/qX1_to_pX1."
  echo "      Current runtime will use p8e1 quire-compatible high-precision fallback."
fi

echo "Built ${SOFTPOSIT_PX1_LIB}"
