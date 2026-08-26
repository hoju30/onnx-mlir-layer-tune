#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate src/Dialect/ONNX/ONNXOps.td.inc and src/Builder/OpBuildTable.inc
# from utils/gen_onnx_mlir.py.
#
# gen_onnx_mlir.py refuses to run against any onnx package version other
# than the exact one it was written against (utils/gen_onnx_mlir.py's
# `current_onnx_version`, kept in sync with the `onnx==` pin in
# requirements.txt). A plain `python3 utils/gen_onnx_mlir.py` will fail if
# the interpreter's already-installed onnx (from some other project, an
# unrelated venv, a newer system package, etc.) doesn't happen to match --
# which is a likely, not a hypothetical, situation on a developer machine
# that works on more than one onnx-touching project.
#
# This script sidesteps that by installing the pinned onnx version into an
# isolated directory with `pip install --target` (no venv/virtualenv needed,
# nothing about the interpreter's normal onnx install is touched or
# upgraded/downgraded) and pointing PYTHONPATH at just that directory for
# the one invocation of gen_onnx_mlir.py. Safe to re-run any time; the
# install step is skipped if that directory already has the right version.
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(dirname -- "$SCRIPT_DIR")
PY="${PYTHON3:-python3}"

ONNX_VERSION=$(grep -E '^onnx==' "$REPO_ROOT/requirements.txt" | head -1 | cut -d= -f3)
if [ -z "$ONNX_VERSION" ]; then
  echo "error: no 'onnx==<version>' line found in $REPO_ROOT/requirements.txt" >&2
  exit 1
fi

ONNX_ENV_DIR="$REPO_ROOT/build/.onnx-codegen-env"
INSTALLED_VERSION=""
if [ -f "$ONNX_ENV_DIR/onnx/version.py" ]; then
  INSTALLED_VERSION=$(PYTHONPATH="$ONNX_ENV_DIR" "$PY" -c "import onnx; print(onnx.__version__)" 2>/dev/null || true)
fi

if [ "$INSTALLED_VERSION" != "$ONNX_VERSION" ]; then
  echo "Installing onnx==$ONNX_VERSION into $ONNX_ENV_DIR (isolated; your"
  echo "interpreter's own onnx install, if any, is left untouched)..."
  mkdir -p "$ONNX_ENV_DIR"
  "$PY" -m pip install --no-deps --target "$ONNX_ENV_DIR" "onnx==$ONNX_VERSION"
else
  echo "onnx==$ONNX_VERSION already present in $ONNX_ENV_DIR, reusing it."
fi

echo "Running gen_onnx_mlir.py..."
PYTHONPATH="$ONNX_ENV_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$SCRIPT_DIR/gen_onnx_mlir.py" "$@"

# Only the default run (no query/dry-run flag) actually writes the two
# generated files -- e.g. --check-operation-version/--list-operation-version
# just print a report and exit, so there's nothing to move in that case.
if [ -f "$SCRIPT_DIR/ONNXOps.td.inc" ] && [ -f "$SCRIPT_DIR/OpBuildTable.inc" ]; then
  mv "$SCRIPT_DIR/ONNXOps.td.inc" "$REPO_ROOT/src/Dialect/ONNX/ONNXOps.td.inc"
  mv "$SCRIPT_DIR/OpBuildTable.inc" "$REPO_ROOT/src/Builder/OpBuildTable.inc"
  # ninja/CMake don't track ONNXOps.td.inc as a dependency of the
  # mlir-tblgen step that consumes it (it's pulled in via a TableGen
  # `include`, which CMake's dependency scanner doesn't see), so a manual
  # regen like this one needs to bump something ninja DOES track to force
  # tblgen to notice the change.
  touch "$REPO_ROOT/src/Dialect/ONNX/ONNX.td"
  echo "Wrote src/Dialect/ONNX/ONNXOps.td.inc and src/Builder/OpBuildTable.inc."
  echo "Rebuild (e.g. 'cmake --build build --target onnx-mlir-opt') to pick up the change."
fi
