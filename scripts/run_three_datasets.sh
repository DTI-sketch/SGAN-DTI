#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-datasets}
DEVICE=${2:-cuda:0}
OUTPUT_ROOT=${3:-outputs}

COMMON_ARGS=(
  --output-dir "${OUTPUT_ROOT}"
  --device "${DEVICE}"
  --seed 3407
  --fold-indices 0,1,2,3,4,5,6,7,8,9
  --epochs 100
  --patience 20
  --weight-decay 2e-4
  --batch-size 128
  --predictor gcn
  --semantic-fusion concatenation
  --threshold-objective f1_mcc
)

python -u run_dti.py \
  --dataset data_luo --data-dir "${DATA_ROOT}/data_luo" \
  --hidden-dim 32 --num-heads 1 --num-layers 3 --lr 5e-4 --dropout 0.5 \
  "${COMMON_ARGS[@]}"

python -u run_dti.py \
  --dataset data --data-dir "${DATA_ROOT}/data" \
  --hidden-dim 128 --num-heads 2 --num-layers 1 --lr 1e-3 --dropout 0.6 \
  "${COMMON_ARGS[@]}"

python -u run_dti.py \
  --dataset data_zeng --data-dir "${DATA_ROOT}/data_zeng" \
  --hidden-dim 64 --num-heads 2 --num-layers 2 --lr 1e-3 --dropout 0.6 \
  "${COMMON_ARGS[@]}"
