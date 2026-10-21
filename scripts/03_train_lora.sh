#!/usr/bin/env bash
# Stage 4: LoRA fine-tuning of Qwen3-VL-2B-Instruct with ms-swift, then merge.
# Config: configs/lora_qwen3vl2b_v1.yaml (r=8, alpha=32, 3 epochs, DeepSpeed ZeRO-2, 2 GPUs, seed 42)
#
# The released Wigtn/Qwen3-VL-2B-WigtnOCR is checkpoint-600 of the v1 run (lowest eval loss),
# not the last checkpoint. Set CKPT to merge a different one.
set -euo pipefail

python -m training.lora_trainer --config configs/lora_qwen3vl2b_v1.yaml

RUN_DIR="$(ls -dt output/wigtnocr-2b-lora/v*-* | head -n 1)"   # newest run
CKPT="${CKPT:-$RUN_DIR/checkpoint-600}"

swift export \
  --adapters "$CKPT" \
  --merge_lora true \
  --output_dir output/wigtnocr-2b-merged
