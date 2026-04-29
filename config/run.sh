#!/usr/bin/env bash
# Usage: run.sh [module] [args...]
# Default: federated_lssvm.train. Threads default to nproc, override with
# OMP_NUM_THREADS=N ./run.sh ... or pass --threads=N to the python module.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -n "${VENV_DIR:-}" ]; then
  :
elif [ -d "$REPO_ROOT/venv" ]; then
  VENV_DIR="$REPO_ROOT/venv"
elif [ -d "$(cd "$REPO_ROOT/.." && pwd)/venv" ]; then
  VENV_DIR="$(cd "$REPO_ROOT/.." && pwd)/venv"
else
  VENV_DIR="$REPO_ROOT/venv"
fi
cd "$REPO_ROOT"
# shellcheck source=/dev/null
[ -f "$VENV_DIR/bin/activate" ] && source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
export FHE_DEFAULT_THREADS=${FHE_DEFAULT_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${FHE_DEFAULT_THREADS}}
export FHE_THREADS=${FHE_THREADS:-${FHE_DEFAULT_THREADS}}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export OMP_PLACES=${OMP_PLACES:-cores}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
module="federated_lssvm.train"
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  module="$1"
  shift
fi
LAST_CPU=$(( $(nproc) - 1 ))
PYTHON_BIN="${VENV_DIR}/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${VENV_DIR}/bin/python3"
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "run.sh: could not find python in $VENV_DIR" >&2
  exit 1
fi
exec /usr/bin/taskset -c 0-${LAST_CPU} "$PYTHON_BIN" -m "$module" "$@"
