#!/usr/bin/env bash
# KoGovDoc-Bench (294 validation pages): parse, then score retrieval and chunk quality.
# Start the model first: bash scripts/serve_student.sh   (served name: wigtnocr-2b, port 8000)
#
#   A. v1 protocol (README "Key Results"): Text NED, BC/CS, Hit@K / MRR / nDCG
#   B. RCPS protocol (EMNLP 2026 paper):   Hit@K / MRR / nDCG / RCPS, Boundary Clarity
# The two protocols use different queries and relevance rules; see the README.
set -euo pipefail

NAME="${NAME:-wigtnocr-2b}"
API_URL="${API_URL:-http://localhost:8000/v1/chat/completions}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR=datasets/kogovdoc-bench
RADP_DIR=third_party/WigtnOCR-RADP
OUT="results/kogovdoc/$NAME"

# 294 page images + val.jsonl (training prompts and reference Markdown per page)
python -c "from huggingface_hub import snapshot_download; snapshot_download('Wigtn/KoGovDoc-Bench', repo_type='dataset', local_dir='$DATA_DIR')"

# Parse every page with the prompts it was trained with, and score Text NED against the reference
python scripts/eval_kogovdoc/run_val_eval.py --val-jsonl "$DATA_DIR/val.jsonl" --image-root "$DATA_DIR/images" \
  --output-dir "$OUT" --model wigtnocr-2b --api-url "$API_URL" --batch-size 8 --name "$NAME"

# ---- A. v1 protocol ---------------------------------------------------------------------------
# Needs two more endpoints, started in other terminals:
#   python scripts/eval_kogovdoc/serve_bge_m3.py --port 8001 --device "$DEVICE"      (embeddings)
#   vllm serve Qwen/Qwen2.5-1.5B-Instruct --served-model-name qwen2.5-1.5b-instruct --port 8020   (perplexity)
# run_retrieval_eval.py reads each parser's outputs from results/kogovdoc/<parser>_val/predictions
# (WigtnOCR-2B: v1_val) and builds its queries from the WigtnOCR-2B outputs in that folder.
python scripts/eval_kogovdoc/run_bccs_eval.py --parsers wigtnocr --pred-dirs "$OUT/predictions" \
  --strategies semantic --output-dir "results/kogovdoc/${NAME}_bccs"
mkdir -p results/kogovdoc/v1_val && ln -sfn "$(pwd)/$OUT/predictions" results/kogovdoc/v1_val/predictions
python scripts/eval_kogovdoc/run_retrieval_eval.py --parsers wigtnocr \
  --output-dir "results/kogovdoc/${NAME}_retrieval_v1"

# ---- B. RCPS protocol -------------------------------------------------------------------------
[[ -d "$RADP_DIR" ]] || git clone --depth 1 https://github.com/wigtn/WigtnOCR-RADP "$RADP_DIR"
python scripts/eval_kogovdoc/run_rcps_eval.py --pred-dir "$OUT/predictions" --val-jsonl "$DATA_DIR/val.jsonl" \
  --qa "$RADP_DIR/data/KoGovDoc-RAG/qa_pairs_v1.jsonl" --device "$DEVICE" \
  --out "results/kogovdoc/${NAME}_rcps.json"
python "$RADP_DIR/scripts/evaluation/compute_parser_bc.py" --parser-dir "$OUT/predictions" --label "$NAME" \
  --model Qwen/Qwen3-VL-2B-Instruct --device "$DEVICE" --max-tokens 1024 --no-references \
  --out "results/kogovdoc/${NAME}_rcps_bc.json"
