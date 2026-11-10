#!/usr/bin/env bash
# ==============================================================================
# SDBC-Mamba: Cityscapes Final Run
# Classes: 19 | Resolution: 768×768 | Backbone: Swin-B (22K pretrained)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "======================================================================"
echo " SDBC-Mamba — Cityscapes"
echo " CUDA Dev  : $CUDA_VISIBLE_DEVICES"
echo " GPU       : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo " PyTorch   : $(python -c 'import torch; print(torch.__version__)')"
echo " Config    : configs/sdbc_mamba_cityscapes.yaml"
echo " Storage   : /DATA1/b23cs1037/cv"
echo "======================================================================"

python train.py \
    --config configs/sdbc_mamba_cityscapes.yaml \
    --exp sdbc_mamba_cityscapes \
    "$@"
