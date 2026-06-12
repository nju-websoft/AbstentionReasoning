#!/usr/bin/env bash
set -euo pipefail

# Stage 1: generate vanilla reasoning trajectories without intervention.
# Outputs are written under ../data/out/stage1/.
#
# Usage examples:
#   bash stage_1_generate_vanilla.sh
#   MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B bash stage_1_generate_vanilla.sh
#   DATASET=UMWP bash stage_1_generate_vanilla.sh
#   RUN_SPLIT=unanswer bash stage_1_generate_vanilla.sh
#
# RUN_SPLIT options:
#   both     run answerable and unanswerable data
#   answer   run answerable data only
#   unanswer run unanswerable data only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"
DATASET="${DATASET:-SUM}"
RUN_SPLIT="${RUN_SPLIT:-both}"

run_answerable() {
  echo "[Stage 1] Running vanilla generation on answerable questions"
  python main_vllm.py \
    --dataset "${DATASET}" \
    --split answer \
    --model_name "${MODEL_PATH}"
}

run_unanswerable() {
  echo "[Stage 1] Running vanilla generation on unanswerable questions"
  python main_vllm.py \
    --dataset "${DATASET}" \
    --split unanswer \
    --model_name "${MODEL_PATH}"
}

case "${RUN_SPLIT}" in
  both)
    run_answerable
    run_unanswerable
    ;;
  answer)
    run_answerable
    ;;
  unanswer)
    run_unanswerable
    ;;
  *)
    echo "Unknown RUN_SPLIT='${RUN_SPLIT}'. Expected: both, answer, or unanswer." >&2
    exit 1
    ;;
esac

echo "[Stage 1] Done."
