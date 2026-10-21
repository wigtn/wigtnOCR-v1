#!/usr/bin/env bash
# Stage 1: teacher pseudo-labels (page image -> Markdown), one file per page.
# Start the teacher first: bash scripts/serve_teacher.sh
#
# Input layout (PDFs are not distributed with this repository):
#   datasets/documents/<doc_id>/doc.pdf     Korean government documents (KoGovDoc)
#   datasets/papers/<doc_id>/paper.pdf      English arXiv papers
# Output: datasets/<dataset>/<doc_id>/gt_pages/page_NNNN.md (pages are rendered at 200 DPI)
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000/v1/chat/completions}"

python scripts/training/generate_pseudo_gt.py --dataset documents --api-url "$API_URL" --model qwen3-vl-30b-thinking
python scripts/training/generate_pseudo_gt.py --dataset papers --api-url "$API_URL" --model qwen3-vl-30b-thinking
