#!/usr/bin/env bash
# OmniDocBench v1.5 (1,355 pages) with the v1 evaluation code:
# Text NED, Table TEDS/TEDS-S, Reading Order NED, skipped pages, then Formula CDM.
# Start the model first: bash scripts/serve_student.sh   (served name: wigtnocr-2b, port 8000)
#
# The v1 numbers were measured on OmniDocBench v1.5. Its main branch has since moved to a larger
# v1.6 set, so the data is pinned to the v1_5 revision here.
set -euo pipefail

NAME="${NAME:-wigtnocr-2b}"
API_URL="${API_URL:-http://localhost:8000/v1/chat/completions}"
DATA_DIR=datasets/omnidocbench

python -c "from huggingface_hub import snapshot_download; snapshot_download('opendatalab/OmniDocBench', repo_type='dataset', revision='v1_5', local_dir='$DATA_DIR')"

# Inference: one Markdown file per page image, English pseudo-label prompts, temperature 0.1
python scripts/eval_omnidocbench/run_inference.py --omnidocbench-dir "$DATA_DIR" \
  --output-dir "results/omnidocbench/$NAME" --model wigtnocr-2b --api-url "$API_URL" \
  --max-tokens 8192 --batch-size 4

# Text / table / reading order, written to results/omnidocbench/$NAME_results.json
python scripts/eval_omnidocbench/run_eval.py --pred-dir "results/omnidocbench/$NAME" --name "$NAME" \
  --gt-path "$DATA_DIR/OmniDocBench.json" --quiet

# Formula CDM F1 (needs Node.js, ImageMagick and pdflatex; see evaluation/metrics/cdm/NOTICE.md)
python scripts/eval_omnidocbench/run_cdm.py --pred-dir "results/omnidocbench/$NAME" --name "$NAME" --pools 8
