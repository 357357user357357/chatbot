#!/usr/bin/env bash
# Install TileScale / TileLang for the tilechat kernels.
#
# Option A (default): the upstream TileLang wheel from PyPI.  TileScale keeps
# TileLang's Python kernel language, compiler, JIT and cache interfaces, so
# ordinary TileLang programs (these kernels included) run on both.
#
# Option B (INSTALL_TILESCALE_SOURCE=1): build tilescale from source for its
# single-host multi-GPU distributed runtime.  Needs a C++ toolchain, CMake
# and git submodules; this is only needed for the distributed features, not
# for the chatbot kernels.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "==> checking CUDA toolkit (needed for JIT compilation)"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "WARNING: nvcc not found in PATH. tilelang needs it to JIT kernels."
  echo "         Debian/Ubuntu: sudo apt install nvidia-cuda-toolkit"
fi

echo "==> installing python dependencies"
$PYTHON -m pip install torch tilelang pytest

if [ "${INSTALL_TILESCALE_SOURCE:-0}" = "1" ]; then
  echo "==> building tilescale from source"
  $PYTHON -m pip install --no-build-isolation "cmake>=3.26" ninja
  rm -rf /tmp/tilescale-src
  git clone --recursive https://github.com/tile-ai/tilescale /tmp/tilescale-src
  $PYTHON -m pip install -v /tmp/tilescale-src
fi

echo "==> verifying installation"
$PYTHON scripts/check_tilescale.py
