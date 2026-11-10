#!/usr/bin/env bash
# ==============================================================================
# SDBC-Mamba: Final Training Run Launch Script (Oxford Pet)
# Dataset: The Oxford-IIIT Pet Dataset (512x512)
# Storage: /DATA1/b23cs1037/cv
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

echo "======================================================================"
echo " SDBC-Mamba — Oxford Pet Final Run"
echo " Workspace : $SCRIPT_DIR"
echo " CUDA Dev  : $CUDA_VISIBLE_DEVICES"
echo " PyTorch   : $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo " GPU       : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo " Config    : configs/sdbc_mamba_pet.yaml"
echo " Storage   : /DATA1/b23cs1037/cv"
echo "======================================================================"

python train.py \
    --config configs/sdbc_mamba_pet.yaml \
    --exp sdbc_mamba_final \
    "$@"
