#!/usr/bin/env bash
# run_training.sh — convenience wrapper for neural-forge training.
#
# Usage:
#   bash run_training.sh [--config <path>] [extra OmegaConf overrides …]
#
# Examples:
#   bash run_training.sh
#   bash run_training.sh --config configs/resnet_cifar10.yaml
#   bash run_training.sh training.epochs=100 training.lr=1e-3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/configs/default.yaml"

# Parse --config argument (everything else is passed through to the trainer)
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "================================================"
echo " Neural Forge — Training"
echo " Config : ${CONFIG}"
echo "================================================"

# Activate virtual environment if present
if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

PYTHONPATH="${SCRIPT_DIR}/src" python -m neural_forge.training.trainer \
    --config "${CONFIG}" \
    "${EXTRA_ARGS[@]}"
