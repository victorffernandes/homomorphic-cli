#!/bin/bash
# Activate unified venv for lwe project
# Run with: source activate_env.sh

source venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
echo "venv active. python: $(which python)"
echo ""
echo "Entry points:"
echo "  python -m federated_lssvm.train       # federated training"
echo "  python -m federated_lssvm.infer       # federated inference"
echo "  python -m lssvm.cipher                # single-client cipher LSSVM"
echo "  python -m lssvm.inference             # encrypted inference"
echo "  python -m lssvm.plain                 # plaintext reference"
echo "  bash config/run.sh [module] [args]    # threaded runner"
