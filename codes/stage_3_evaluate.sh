#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-SUM}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-Distill-Llama-8B}"
STAGE2_DIR="${STAGE2_DIR:-../data/out/stage2/${DATASET}/${MODEL_NAME}}"
UNSOLVE_PATH="${UNSOLVE_PATH:-${STAGE2_DIR}/unsolve_intervention_result.jsonl}"
SOLVE_PATH="${SOLVE_PATH:-${STAGE2_DIR}/solve_intervention_result.jsonl}"
ANSWER_JUDGE="${ANSWER_JUDGE:-exact}"
OUTPUT_PATH="${OUTPUT_PATH:-${STAGE2_DIR}/stage3_metrics.json}"

"$PYTHON_BIN" stage_3_evaluate.py \
  --unsolve_path "$UNSOLVE_PATH" \
  --solve_path "$SOLVE_PATH" \
  --answer_judge "$ANSWER_JUDGE" \
  --output_path "$OUTPUT_PATH"
