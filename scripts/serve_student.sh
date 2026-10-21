#!/usr/bin/env bash
# WigtnOCR-2B (or any Qwen3-VL checkpoint) on vLLM for evaluation.
#   MODEL=Wigtn/Qwen3-VL-2B-WigtnOCR    released model (default)
#   MODEL=output/wigtnocr-2b-merged     your own merged checkpoint (scripts/03_train_lora.sh)
#   MODEL=Qwen/Qwen3-VL-2B-Instruct     the untuned base, for comparison
set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-Wigtn/Qwen3-VL-2B-WigtnOCR}"

exec vllm serve "$MODEL" \
  --served-model-name wigtnocr-2b \
  --max-model-len 16384 \
  --port "$PORT"
