#!/usr/bin/env bash
# Usage: run.sh [script.py] [args...]
# Default script: lssvm_cipher.py. Threads default to nproc, override with
# OMP_NUM_THREADS=N ./run.sh ... or pass --threads=N to the python script.
set -euo pipefail
cd "/home/main/Documents/Projects/lwe/open-fhe"
source "/home/main/Documents/Projects/lwe/venv/bin/activate"
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
export FHE_DEFAULT_THREADS=${FHE_DEFAULT_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${FHE_DEFAULT_THREADS}}
export FHE_THREADS=${FHE_THREADS:-${FHE_DEFAULT_THREADS}}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export OMP_PLACES=${OMP_PLACES:-cores}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
script="federated_lssvm.py"
if [ $# -gt 0 ] && [[ "$1" == *.py ]]; then
  script="$1"
  shift
fi
# Launch Python with unrestricted CPU affinity (all cores 0-N)
LAST_CPU=$(( $(nproc) - 1 ))
exec /usr/bin/taskset -c 0-${LAST_CPU} python "$script" "$@"
