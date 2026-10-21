#!/usr/bin/env bash
# Judge for pseudo-label quality: Qwen3.5-122B-A10B (text only) on vLLM, two GPUs.
# The v1 run served an NVFP4 build of Qwen3.5-122B-A10B with a nightly vLLM, TP=2,
# gpu_memory_utilization 0.80 and the deepseek_r1 reasoning parser, which moves the
# model's thinking out of the response content. Which NVFP4 repository was used is not
# recorded; nvidia/Qwen3.5-122B-A10B-NVFP4 is the public NVFP4 build (NVFP4 needs a
# Blackwell GPU). Set JUDGE_MODEL=Qwen/Qwen3.5-122B-A10B for the original weights.
set -euo pipefail

PORT="${PORT:-8000}"
JUDGE_MODEL="${JUDGE_MODEL:-nvidia/Qwen3.5-122B-A10B-NVFP4}"

exec vllm serve "$JUDGE_MODEL" \
  --served-model-name qwen3.5-122b \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.80 \
  --reasoning-parser deepseek_r1 \
  --port "$PORT"
