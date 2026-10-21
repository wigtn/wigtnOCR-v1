#!/usr/bin/env bash
# Teacher for pseudo-labels: Qwen3-VL-30B-A3B-Thinking on vLLM, two GPUs.
# Settings follow the v1 run (vLLM 0.13.0, TP=2, gpu_memory_utilization 0.95, max_model_len 16384).
#
# The v1 run also loaded a custom chat template (qwen3-vl-no-think.jinja) that suppressed
# thinking text at generation time. That template is not in this repository. Pass your own
# with CHAT_TEMPLATE=path/to/template.jinja; without it, training/gt_generator.py still
# strips <think> blocks after generation, but pseudo-labels can differ from the v1 set.
set -euo pipefail

PORT="${PORT:-8000}"
args=(
  --served-model-name qwen3-vl-30b-thinking
  --tensor-parallel-size 2
  --gpu-memory-utilization 0.95
  --max-model-len 16384
  --port "$PORT"
)
if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
  args+=(--chat-template "$CHAT_TEMPLATE")
fi

exec vllm serve Qwen/Qwen3-VL-30B-A3B-Thinking "${args[@]}"
