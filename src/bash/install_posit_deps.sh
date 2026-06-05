#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"
deps_root="${src_root}/.deps"

universal_repo="${deps_root}/universal"
softposit_repo="${deps_root}/SoftPosit"
softposit_lib_dir="${deps_root}/softposit-px1"

universal_url="${UNIVERSAL_GIT_URL:-https://github.com/stillwater-sc/universal.git}"
softposit_url="${SOFTPOSIT_GIT_URL:-https://gitlab.com/cerlane/SoftPosit.git}"
jobs="${POSIT_INSTALL_JOBS:-8}"
reclone=0

usage() {
  cat <<'EOF'
Usage:
  install_posit_deps.sh [--reclone] [--jobs N]

What it does:
  1. Installs/refreshes the header-only Universal library into src/.deps/universal
  2. Installs/refreshes SoftPosit into src/.deps/SoftPosit
  3. Applies a GCC-13-compatible SoftPosit header fix
  4. Builds libsoftposit.a into src/.deps/softposit-px1
EOF
}

clone_or_refresh() {
  local url="$1"
  local dest="$2"

  if [[ "${reclone}" -eq 1 && -d "${dest}" ]]; then
    rm -rf "${dest}"
  fi

  if [[ -d "${dest}/.git" ]]; then
    git -C "${dest}" fetch --depth 1 origin
    git -C "${dest}" reset --hard FETCH_HEAD
  else
    git clone --depth 1 "${url}" "${dest}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reclone)
      reclone=1
      shift
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "ERROR: --jobs needs a value"; exit 2; }
      jobs="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg: $1"
      usage
      exit 2
      ;;
  esac
done

mkdir -p "${deps_root}" "${softposit_lib_dir}"

clone_or_refresh "${universal_url}" "${universal_repo}"
clone_or_refresh "${softposit_url}" "${softposit_repo}"

softposit_types="${softposit_repo}/source/include/softposit_types.h"
if [[ ! -f "${softposit_types}" ]]; then
  echo "ERROR: missing ${softposit_types}"
  exit 2
fi

# GCC 13 rejects these C++-style union member initializers when building the C library.
perl -0pi -e 's/uint32_t ui=0;/uint32_t ui;/g; s/uint64_t ui\[2\]=\{0,0\};/uint64_t ui[2];/g; s/uint64_t ui\[8\]=\{0,0,0,0, 0,0,0,0\};/uint64_t ui[8];/g;' "${softposit_types}"

make -C "${softposit_repo}/build/Linux-x86_64-GCC" -j"${jobs}" all
ln -sf "${softposit_repo}/build/Linux-x86_64-GCC/softposit.a" "${softposit_lib_dir}/libsoftposit.a"

cat <<EOF
Installed posit deps:
  universal headers : ${universal_repo}/include/sw
  softposit root    : ${softposit_repo}
  softposit library : ${softposit_lib_dir}/libsoftposit.a

Useful overrides:
  export UNIVERSAL_INCLUDE_DIR=${universal_repo}/include/sw
  export SOFTPOSIT_ROOT=${softposit_repo}
  export SOFTPOSIT_INCLUDE_DIR=${softposit_repo}/source/include
  export SOFTPOSIT_LIB_PATH=${softposit_lib_dir}/libsoftposit.a
EOF
