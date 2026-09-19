#!/usr/bin/env bash
# One-shot setup + validation for the CMP 50HX (TU102, Turing, sm_75, 10 GB).
#
# Safe to re-run (idempotent). Never uses sudo: if a system-level piece is
# missing (driver, nvcc) it prints the exact commands and stops.
#
#   bash scripts/install_50hx.sh
#
# Env overrides: PYTHON, VENV_DIR, TORCH_INDEX (default cu130 wheel index).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

banner() { printf '\n==> %s\n' "$1"; }

# ---------------------------------------------------------------- 1. driver
banner "1/6 GPU / driver check"
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: nvidia-smi not found. Install the NVIDIA driver first, e.g."
  echo "  sudo apt-get install nvidia-driver-580   # or newer"
  echo "then reboot and re-run this script."
  exit 1
}
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
GOOD=""; BAD=""
while IFS=, read -r idx cc; do
  cc="${cc//[[:space:]]/}"
  if [ "$((10#${cc%%.*} * 10 + 10#${cc#*.}))" -ge 75 ]; then
    GOOD="${GOOD:+$GOOD,}$idx"
  else
    BAD="${BAD:+$BAD,}$idx"
  fi
done < <(nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader)
[ -n "$GOOD" ] || {
  echo "ERROR: no CUDA device with compute capability >= 7.5. Modern torch"
  echo "(cu12.8+/cu13) wheels dropped sm_50..sm_70 — those GPUs cannot run"
  echo "these kernels."
  exit 1
}
if [ -n "$BAD" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "NOTE: hiding device(s) [$BAD] (compute cap < 7.5) via"
  echo "      CUDA_VISIBLE_DEVICES=$GOOD -- cuDNN >= 9.11 refuses to run in a"
  echo "      process that can see an older GPU, which breaks the tutorial"
  echo "      encoder's cuDNN GRU. Export it in your shell too."
  export CUDA_VISIBLE_DEVICES="$GOOD"
fi
echo "using CUDA device(s): ${CUDA_VISIBLE_DEVICES:-all visible}"

# ------------------------------------------------------------- 2. nvcc
banner "2/6 CUDA toolkit (nvcc) check"
if ! command -v nvcc >/dev/null 2>&1; then
  . /etc/os-release 2>/dev/null || VERSION_ID="24.04"
  UB="ubuntu${VERSION_ID%%.*}04"
  echo "ERROR: nvcc not in PATH — tilelang JIT-compiles kernels at runtime."
  echo "Install the CUDA toolkit (network repo, ${UB}):"
  echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/${UB}/x86_64/cuda-keyring_1.1-1_all.deb"
  echo "  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update"
  echo "  sudo apt-get -y install cuda-toolkit          # CUDA 13.x meta package"
  echo "  export PATH=/usr/local/cuda/bin:\$PATH         # add to ~/.bashrc"
  echo "then re-run this script."
  exit 1
fi
nvcc --version | tail -n1

# ---------------------------------------------------------- 3. venv + deps
banner "3/6 python environment"
if [ -z "${VIRTUAL_ENV:-}" ] && [ ! -x "$VENV_DIR/bin/python" ]; then
  $PYTHON -m venv "$VENV_DIR"
  echo "created $VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || PY="$PYTHON"   # already inside an activated venv
echo "using python: $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements.txt

# torch from PyPI should be the CUDA build on Linux; verify and fix if not.
if ! "$PY" - <<'EOF'
import sys
import torch
if not torch.version.cuda:
    sys.exit(f"torch {torch.__version__} has no CUDA build")
print(f"torch {torch.__version__} (cuda {torch.version.cuda})")
EOF
then
  echo "==> torch lacks a CUDA build — reinstalling from $TORCH_INDEX"
  "$PY" -m pip install -q --force-reinstall torch --index-url "$TORCH_INDEX"
fi

# --------------------------------------------- 4. nvcc <-> host compiler probe
# tilelang JIT-invokes `nvcc -ccbin=<c++ from CXX/CC or PATH>`, and nvcc
# rejects host compilers newer than it supports ("#error -- unsupported GNU
# version!").  tilechat probes exactly that command before its first JIT and
# auto-switches to an older /usr/bin/g++-N when needed; run the same probe
# here so the decision is visible (and fatal) at install time.
banner "4/6 nvcc <-> host compiler probe (tilelang-style)"
if ! "$PY" - <<'EOF'
import sys
from tilechat import kernels
kernels._ensure_jit_host_compiler()
note = kernels.host_compiler_note  # read via the module: it is set by the call
print("host compiler :", note or "default")
sys.exit(1 if note.startswith("no nvcc-compatible") else 0)
EOF
then
  echo "ERROR: no C++ host compiler in PATH is accepted by this nvcc."
  echo "Known case: nvcc 12.4 rejects gcc > 13. Fixes:"
  echo "  a) sudo apt-get install g++-13"
  echo "     export CXX=/usr/bin/g++-13    # add to ~/.bashrc for other CUDA work"
  echo "  b) newer toolkit: install CUDA 13.x as shown in step 2."
  exit 1
fi

# ------------------------------------------------------------ 5. validate
banner "5/6 kernel validation (compiles for sm_75, checks vs references)"
"$PY" scripts/check_tilescale.py --bench

banner "6/6 test-suite + strict-backend check"
"$PY" -m pytest tests/ -q | tail -n 3
SUMMARY="$("$PY" -m pytest tests/ -q -p no:cacheprovider 2>/dev/null | tail -n 1 || true)"
case "$SUMMARY" in
  *" skipped"*) echo "NOTE: some tests were skipped — GPU kernel tests should"
                echo "NOT skip here. Re-run 'python scripts/check_tilescale.py' and read why." ;;
esac
TILECHAT_BACKEND=tilescale "$PY" -m tilechat check

cat <<'EOF'

All green. Train with the TileLang kernels:

  source .venv/bin/activate
  python -m tilechat train

(First iterations include one-time JIT compiles for sm_75; cached after.)
EOF