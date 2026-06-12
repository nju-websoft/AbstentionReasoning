#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-SUM}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-Distill-Llama-8B}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
PROBE_LAYER="${PROBE_LAYER:-22}"
THEROD="${THEROD:-0.4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../data/out/stage2}"

"$PYTHON_BIN" a_inter.py \
  --dataset "$DATASET" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --probe_layer "$PROBE_LAYER" \
  --therod "$THEROD" \
  --output_root "$OUTPUT_ROOT"
