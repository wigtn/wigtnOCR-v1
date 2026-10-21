#!/usr/bin/env bash
# Stage 2: judge scoring of pseudo-labels. Start the judge first: bash scripts/serve_judge.sh
# Stage 3: filtering (score >= 3), per-document cap (max_doc_ratio 0.25), 90/10 split.
#
# Sampling follows the v1 run: 30% of KoGovDoc pages, 100% of arXiv pages, seed 42.
# Pages that were not sampled enter training without a judge score.
# Rubric and bias correction: docs/pseudo-label-judge.md
#
# After this script the v1 run had 2,985 samples. A manual cleaning pass then removed
# leftover reasoning text from 283 samples and dropped 24, giving 2,667 train / 294 val.
# That pass is not scripted here.
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000/v1/chat/completions}"

python scripts/training/validate_gt.py --dataset documents --sample-ratio 0.3 --seed 42 --api-url "$API_URL" --model qwen3.5-122b
python scripts/training/validate_gt.py --dataset papers --sample-ratio 1.0 --seed 42 --api-url "$API_URL" --model qwen3.5-122b

python scripts/training/prepare_training_data.py --dataset all \
  --min-score 3 --max-doc-ratio 0.25 --val-ratio 0.1 --seed 42 --dpi 200
