# Fine-tuning Qwen3-VL for document parsing with LoRA

[English](qwen3-vl-lora-guide.md) | [한국어](qwen3-vl-lora-guide.ko.md)

This guide walks through the pipeline that produced
[`Wigtn/Qwen3-VL-2B-WigtnOCR`](https://huggingface.co/Wigtn/Qwen3-VL-2B-WigtnOCR): a teacher model writes
Markdown for each page, a judge model filters it, and Qwen3-VL-2B-Instruct is fine-tuned on the result with LoRA.
Every command below is one of the scripts in this repository, run from the repository root. The last section
lists how each one was checked.

WigtnOCR is an independent project built on openly released Qwen models. It is not an official Qwen release.

## 0. Hardware and environments

The v1 run used one machine with two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (98 GB each), CUDA 12.8.
The teacher, the judge and the training job used the same two GPUs one after another, never at the same time.

| Step | Model | GPUs | Recorded cost in the v1 run |
|---|---|---|---|
| Pseudo-labels | Qwen3-VL-30B-A3B-Thinking | 2 (tensor parallel) | 4,501 pages |
| Judge | Qwen3.5-122B-A10B, NVFP4 | 2 (tensor parallel) | 7.8 h for 1,047 KoGovDoc pages, 6.2 h for 864 arXiv pages |
| LoRA training | Qwen3-VL-2B-Instruct | 2 (DeepSpeed ZeRO-2) | 31 min 37 s, 1,002 steps, 47.25 GiB peak per GPU |

Use three Python environments, because vLLM and ms-swift pin different library versions:

```bash
git clone https://github.com/wigtn/wigtnOCR-v1 && cd wigtnOCR-v1
python -m venv .venv-train && .venv-train/bin/pip install -r requirements.txt        # ms-swift 4.0.1, torch 2.9.1, transformers 4.57.3
python -m venv .venv-serve && .venv-serve/bin/pip install -r requirements-serve.txt  # vLLM 0.13.0
python -m venv .venv-eval  && .venv-eval/bin/pip install -r requirements-eval.txt    # evaluation code + RCPS reference implementation
```

Activate the matching environment before each step (`source .venv-train/bin/activate`, and so on).

## 1. Lay out your documents

```
datasets/
├── documents/<doc_id>/doc.pdf     # Korean pages: Korean teacher prompt
└── papers/<doc_id>/paper.pdf      # English pages: English teacher prompt
```

The folder name picks the prompt language. The PDFs used for WigtnOCR are not distributed here.

## 2. Pseudo-labels from the teacher

```bash
bash scripts/serve_teacher.sh          # terminal 1, .venv-serve
bash scripts/01_pseudo_labels.sh       # terminal 2, .venv-train
```

Each page is rendered at 200 DPI and sent to the teacher with temperature 0.1 and up to 8,192 output tokens.
The result is one file per page: `datasets/<dataset>/<doc_id>/gt_pages/page_NNNN.md`. Finished pages are
skipped, so the script can be restarted.

Try one document first:

```bash
python scripts/training/generate_pseudo_gt.py --dataset documents --doc-id <doc_id> --max-pages 3
```

The teacher is a thinking model. `training/gt_generator.py` removes `<think>` blocks and leading reasoning text
from every response (`_clean_response`). The v1 run also used a custom chat template that suppressed thinking at
generation time. That template is not in this repository; pass your own with `CHAT_TEMPLATE=... bash scripts/serve_teacher.sh`.

## 3. Judge, filter and split

```bash
bash scripts/serve_judge.sh            # terminal 1, .venv-serve (stop the teacher first)
bash scripts/02_judge_filter.sh        # terminal 2, .venv-train
```

The judge scores each sampled page from the Markdown alone, on five 1–5 dimensions plus an overall score.
Pages with an overall score below 3 are removed, any document above 25% of its dataset is down-sampled, and
the rest is split 90/10 with seed 42. The rubric, the sampling rates and the v1 numbers are in
[`docs/pseudo-label-judge.md`](pseudo-label-judge.md).

Check the counts before rendering any images:

```bash
python scripts/training/prepare_training_data.py --dataset all --dry-run
```

The script writes `datasets/training/train.jsonl` and `val.jsonl` in the ms-swift format, one page per line:

```json
{"messages": [{"role": "system", "content": "You are WigtnOCR, a specialized document parser ..."},
              {"role": "user", "content": "<image>Convert this document page image to well-structured Markdown.\nOutput ONLY the Markdown content. No explanations, no commentary."},
              {"role": "assistant", "content": "<pseudo-label Markdown>"}],
 "images": ["/abs/path/datasets/training/images/documents/<doc_id>/page_NNNN.png"]}
```

The v1 run ended with 2,667 training and 294 validation pages after a manual cleaning pass that this
repository does not script (283 samples cleaned, 24 dropped).

## 4. LoRA fine-tuning

```bash
python -m training.lora_trainer --config configs/lora_qwen3vl2b_v1.yaml --dry-run   # prints the swift command
bash scripts/03_train_lora.sh                                                       # trains, then merges
```

[`configs/lora_qwen3vl2b_v1.yaml`](../configs/lora_qwen3vl2b_v1.yaml) holds the v1 settings. Every value
matches the `args.json` that ms-swift wrote for the v1 run.

| Setting | Value | Note |
|---|---|---|
| Base | `Qwen/Qwen3-VL-2B-Instruct` @ `89644892` | |
| LoRA | r=8, alpha=32, dropout 0.05, `all-linear` | Language-model linear layers only |
| Frozen | vision encoder and vision-language merger | `freeze_vit`, `freeze_aligner` |
| Schedule | 3 epochs, lr 1e-4, cosine, warmup 0.05, weight decay 0.1 | |
| Batch | 1 per GPU × 4 accumulation × 2 GPUs = 8 | |
| Length | 4,096 tokens; longer samples are dropped | `IMAGE_MAX_TOKEN_NUM=1024` caps visual tokens per page |
| Seeds | `seed=42`, `data_seed=42` | Sampling, split and training all use 42 |
| Parallelism | DeepSpeed ZeRO-2, bf16, gradient checkpointing | |

Evaluation loss every 100 steps in the v1 run:

| Step | 100 | 200 | 300 | 400 | 500 | **600** | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Epoch | 0.3 | 0.6 | 0.9 | 1.2 | 1.5 | **1.8** | 2.1 | 2.4 | 2.7 | 3.0 |
| Eval loss | 0.1145 | 0.1037 | 0.0985 | 0.0956 | 0.0936 | **0.0915** | 0.0921 | 0.0920 | 0.0917 | 0.0915 |

The loss stops improving after step 600, and the released model is `checkpoint-600`. ms-swift does not
switch to the best checkpoint by itself in this configuration, so `scripts/03_train_lora.sh` merges
`checkpoint-600` explicitly:

```bash
swift export --adapters output/wigtnocr-2b-lora/<run>/checkpoint-600 --merge_lora true --output_dir output/wigtnocr-2b-merged
```

A wider adapter was also trained: rank 32 for 5 epochs. On OmniDocBench it scored lower on tables than the
released model (Table TEDS 0.610 vs 0.649) and higher on text edit distance (0.306 vs 0.288), while skipping
fewer pages (0 vs 79).

## 5. Serve and evaluate

```bash
MODEL=output/wigtnocr-2b-merged bash scripts/serve_student.sh   # terminal 1, .venv-serve; omit MODEL for the released model
bash scripts/04_eval_omnidocbench.sh                            # terminal 2, .venv-eval
bash scripts/05_eval_kogovdoc.sh                                # terminal 2, .venv-eval
```

Both scripts call the evaluation code that produced the v1 tables (`evaluation/`, `scripts/eval_omnidocbench/`,
`scripts/eval_kogovdoc/`), unchanged apart from portable paths.

- **`04_eval_omnidocbench.sh`** downloads OmniDocBench v1.5 (1,355 pages) and parses every page with the
  English pseudo-label prompts at temperature 0.1. It scores Text NED, Table TEDS, Reading Order NED and
  skipped pages, then Formula CDM. CDM needs Node.js, ImageMagick and `pdflatex`.
- **`05_eval_kogovdoc.sh`** downloads the 294 KoGovDoc-Bench pages, parses them with the prompts the model was
  trained with, and scores Text NED against the reference Markdown. It then scores retrieval and chunk
  quality under two protocols:
  - **v1 protocol** (README "Key Results"): BC/CS with a Qwen2.5-1.5B-Instruct perplexity model, and Hit@K /
    MRR / nDCG with BGE-M3 over semantic chunks. This needs two more endpoints, listed in the script.
  - **RCPS protocol** (EMNLP 2026 paper): 663 Q–A pairs, three retrievers, k in {1, 5, 10}, and Boundary
    Clarity under Qwen3-VL-2B-Instruct. For the released model the paper reports RCPS 0.583, Hit@1 0.549
    and BC 0.610.

## 6. Adapting the recipe to other documents

- **Prompts.** `VLM_TRAINING_SYSTEM_PROMPT` in `training/prompts/templates.py` describes Korean government
  documents. Rewrite it for your domain before step 3, and keep the same prompts for training and inference.
- **Judge sampling.** `--sample-ratio` trades judge time for coverage. Unsampled pages enter training unjudged.
- **Imbalance.** Lower `--max-doc-ratio` if a few documents dominate your page count.
- **Evaluation.** Score candidates on retrieval from your own documents, not only on parsing benchmarks. In the
  RCPS paper, the parser with the best intrinsic boundary score retrieved worst; see
  [wigtn/WigtnOCR-RADP](https://github.com/wigtn/WigtnOCR-RADP).

## How these commands were checked

The full pipeline last ran end to end on the v1 hardware in March 2026. For this release, each command was
checked without that hardware, on the outputs the v1 run stored:

| Command | Check |
|---|---|
| `python -m training.lora_trainer ... --dry-run` | The printed `swift sft` arguments match the v1 `args.json` on every hyperparameter |
| `prepare_training_data.py --dry-run` | On the v1 arXiv pseudo-labels and judge report it yields 646 pages, the v1 count |
| `swift sft`, `swift export`, `vllm serve` flags | Each flag exists in ms-swift 4.0.1 and vLLM 0.13.0 |
| `run_eval.py` (OmniDocBench) | On the released model's 1,276 stored outputs and the v1.5 ground truth: Text NED 0.2884, Table TEDS 0.6490, Reading Order 0.2111, 79 skipped pages, matching the v1 result file |
| `run_val_eval.py --eval-only` (KoGovDoc) | Text NED 0.2852 over 289 of 294 pages, matching the v1 result file |
| `run_retrieval_eval.py` (KoGovDoc v1) | Rebuilds the same 564 queries from the stored WigtnOCR-2B outputs. With a different BGE-M3 server (`serve_bge_m3.py`) the semantic chunk boundaries move, so Hit@K and MRR do not match the v1 table exactly |
| `run_rcps_eval.py` | BGE-M3 on CPU: Hit@1 0.5445, MRR@10 0.5928, nDCG@10 0.6152, identical to the RCPS run |
| `compute_parser_bc.py` (Boundary Clarity) | On an Apple-silicon GPU (`--device mps`): 1,161 boundaries, mean BC 0.6100; the paper reports 0.610 |

Not rerun for this release: the GPU steps (`vllm serve`, `swift sft`, `swift export`, model inference),
Formula CDM, and the v1 BC/CS script, which needs a Qwen2.5-1.5B-Instruct perplexity server.
