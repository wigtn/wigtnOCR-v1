# LoRA로 문서 파싱용 Qwen3-VL fine-tuning하기

[English](qwen3-vl-lora-guide.md) | [한국어](qwen3-vl-lora-guide.ko.md)

이 가이드는
[`Wigtn/Qwen3-VL-2B-WigtnOCR`](https://huggingface.co/Wigtn/Qwen3-VL-2B-WigtnOCR)을 만든 파이프라인을 차례로 설명합니다: teacher 모델이 페이지마다
Markdown을 작성하고 judge 모델이 이를 걸러 낸 뒤 그 결과로 Qwen3-VL-2B-Instruct를 LoRA로 fine-tuning합니다.
아래의 모든 명령은 이 저장소에 있는 스크립트이며 저장소 루트에서 실행합니다. 마지막 절에서
각 명령을 어떻게 확인했는지 정리합니다.

WigtnOCR은 공개된 Qwen 모델을 바탕으로 만든 독립 프로젝트입니다. Qwen 공식 릴리스가 아닙니다.

## 0. 하드웨어와 환경

v1 실행에는 NVIDIA RTX PRO 6000 Blackwell Server Edition GPU 두 장(각 98 GB)과 CUDA 12.8을 갖춘 머신 한 대를 사용했습니다.
teacher, judge, 학습 작업은 같은 GPU 두 장을 차례로 사용했으며 동시에 사용한 적은 없습니다.

| 단계 | 모델 | GPU | v1 실행에서 기록된 비용 |
|---|---|---|---|
| Pseudo-label | Qwen3-VL-30B-A3B-Thinking | 2 (tensor parallel) | 4,501 페이지 |
| Judge | Qwen3.5-122B-A10B, NVFP4 | 2 (tensor parallel) | KoGovDoc 1,047 페이지에 7.8시간, arXiv 864 페이지에 6.2시간 |
| LoRA 학습 | Qwen3-VL-2B-Instruct | 2 (DeepSpeed ZeRO-2) | 31분 37초, 1,002 step, GPU당 최대 47.25 GiB |

vLLM과 ms-swift가 서로 다른 라이브러리 버전을 고정하므로 Python 환경을 세 개 사용합니다:

```bash
git clone https://github.com/wigtn/wigtnOCR-v1 && cd wigtnOCR-v1
python -m venv .venv-train && .venv-train/bin/pip install -r requirements.txt        # ms-swift 4.0.1, torch 2.9.1, transformers 4.57.3
python -m venv .venv-serve && .venv-serve/bin/pip install -r requirements-serve.txt  # vLLM 0.13.0
python -m venv .venv-eval  && .venv-eval/bin/pip install -r requirements-eval.txt    # evaluation code + RCPS reference implementation
```

각 단계 전에 해당 환경을 활성화합니다(`source .venv-train/bin/activate` 등).

## 1. 문서 배치

```
datasets/
├── documents/<doc_id>/doc.pdf     # Korean pages: Korean teacher prompt
└── papers/<doc_id>/paper.pdf      # English pages: English teacher prompt
```

폴더 이름에 따라 프롬프트 언어가 정해집니다. WigtnOCR에 사용한 PDF는 여기서 배포하지 않습니다.

## 2. teacher로 pseudo-label 만들기

```bash
bash scripts/serve_teacher.sh          # terminal 1, .venv-serve
bash scripts/01_pseudo_labels.sh       # terminal 2, .venv-train
```

각 페이지는 200 DPI로 렌더링되어 temperature 0.1, 최대 출력 토큰 8,192개 설정으로 teacher에 전달됩니다.
결과는 페이지마다 파일 하나입니다: `datasets/<dataset>/<doc_id>/gt_pages/page_NNNN.md`. 이미 끝난 페이지는
건너뛰므로 스크립트를 다시 시작할 수 있습니다.

먼저 문서 하나로 시험해 봅니다:

```bash
python scripts/training/generate_pseudo_gt.py --dataset documents --doc-id <doc_id> --max-pages 3
```

teacher는 thinking 모델입니다. `training/gt_generator.py`는 모든 응답에서 `<think>` 블록과 앞부분의 추론 텍스트를
제거합니다(`_clean_response`). v1 실행에서는 생성 단계에서 thinking을 억제하는 사용자 지정 chat template도 사용했습니다.
이 template은 이 저장소에 없습니다. `CHAT_TEMPLATE=... bash scripts/serve_teacher.sh`로 직접 준비한 template을 넘깁니다.

## 3. Judge 채점, 필터링, 분할

```bash
bash scripts/serve_judge.sh            # terminal 1, .venv-serve (stop the teacher first)
bash scripts/02_judge_filter.sh        # terminal 2, .venv-train
```

judge는 샘플링한 각 페이지를 Markdown만 보고 채점합니다. 1–5점 차원 다섯 개와 종합 점수를 매깁니다.
종합 점수가 3 미만인 페이지는 제거하고 데이터셋의 25%를 넘는 문서는 다운샘플링하며
나머지는 seed 42로 90/10 분할합니다. 채점 기준, 샘플링 비율, v1 수치는
[`docs/pseudo-label-judge.md`](pseudo-label-judge.md)에 있습니다.

이미지를 렌더링하기 전에 개수를 확인합니다:

```bash
python scripts/training/prepare_training_data.py --dataset all --dry-run
```

스크립트는 `datasets/training/train.jsonl`과 `val.jsonl`을 ms-swift 형식으로 씁니다. 한 줄에 페이지 하나입니다:

```json
{"messages": [{"role": "system", "content": "You are WigtnOCR, a specialized document parser ..."},
              {"role": "user", "content": "<image>Convert this document page image to well-structured Markdown.\nOutput ONLY the Markdown content. No explanations, no commentary."},
              {"role": "assistant", "content": "<pseudo-label Markdown>"}],
 "images": ["/abs/path/datasets/training/images/documents/<doc_id>/page_NNNN.png"]}
```

v1 실행은 이 저장소에 스크립트로 들어 있지 않은 수동 정리 과정을 거쳐 학습 페이지 2,667개와 검증 페이지 294개로
끝났습니다(샘플 283개 정리, 24개 제외).

## 4. LoRA fine-tuning

```bash
python -m training.lora_trainer --config configs/lora_qwen3vl2b_v1.yaml --dry-run   # prints the swift command
bash scripts/03_train_lora.sh                                                       # trains, then merges
```

[`configs/lora_qwen3vl2b_v1.yaml`](../configs/lora_qwen3vl2b_v1.yaml)에 v1 설정이 들어 있습니다. 모든 값이
v1 실행에서 ms-swift가 기록한 `args.json`과 일치합니다.

| 설정 | 값 | 비고 |
|---|---|---|
| 베이스 모델 | `Qwen/Qwen3-VL-2B-Instruct` @ `89644892` | |
| LoRA | r=8, alpha=32, dropout 0.05, `all-linear` | 언어 모델의 linear 층만 |
| 고정 | vision encoder와 vision-language merger | `freeze_vit`, `freeze_aligner` |
| 스케줄 | 3 epoch, lr 1e-4, cosine, warmup 0.05, weight decay 0.1 | |
| 배치 | GPU당 1 × accumulation 4 × GPU 2개 = 8 | |
| 길이 | 4,096 토큰, 더 긴 샘플은 제외 | `IMAGE_MAX_TOKEN_NUM=1024`가 페이지당 시각 토큰 수를 제한 |
| Seed | `seed=42`, `data_seed=42` | 샘플링, 분할, 학습 모두 42 사용 |
| 병렬화 | DeepSpeed ZeRO-2, bf16, gradient checkpointing | |

v1 실행에서 100 step마다 기록한 evaluation loss:

| Step | 100 | 200 | 300 | 400 | 500 | **600** | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Epoch | 0.3 | 0.6 | 0.9 | 1.2 | 1.5 | **1.8** | 2.1 | 2.4 | 2.7 | 3.0 |
| Eval loss | 0.1145 | 0.1037 | 0.0985 | 0.0956 | 0.0936 | **0.0915** | 0.0921 | 0.0920 | 0.0917 | 0.0915 |

loss는 step 600 이후 더 나아지지 않으며 공개 모델은 `checkpoint-600`입니다. 이 설정에서 ms-swift는
가장 좋은 체크포인트로 스스로 바꾸지 않습니다. 그래서 `scripts/03_train_lora.sh`가
`checkpoint-600`을 명시적으로 병합합니다:

```bash
swift export --adapters output/wigtnocr-2b-lora/<run>/checkpoint-600 --merge_lora true --output_dir output/wigtnocr-2b-merged
```

더 넓은 어댑터도 학습했습니다: rank 32, 5 epoch. OmniDocBench에서 이 어댑터는 공개 모델보다 표 점수가
낮았고(Table TEDS 0.610 vs 0.649) text edit distance는 높았으며(0.306 vs 0.288) 건너뛴 페이지는
더 적었습니다(0 vs 79).

## 5. 서빙과 평가

```bash
MODEL=output/wigtnocr-2b-merged bash scripts/serve_student.sh   # terminal 1, .venv-serve; omit MODEL for the released model
bash scripts/04_eval_omnidocbench.sh                            # terminal 2, .venv-eval
bash scripts/05_eval_kogovdoc.sh                                # terminal 2, .venv-eval
```

두 스크립트는 v1 표를 만든 평가 코드(`evaluation/`, `scripts/eval_omnidocbench/`,
`scripts/eval_kogovdoc/`)를 호출합니다. 이 코드는 경로를 이식 가능하게 바꾼 것 말고는 그대로입니다.

- <strong>`04_eval_omnidocbench.sh`</strong>는 OmniDocBench v1.5(1,355 페이지)를 내려받아 영어 pseudo-label
  프롬프트와 temperature 0.1로 모든 페이지를 파싱합니다. Text NED, Table TEDS, Reading Order NED, 건너뛴
  페이지를 채점한 다음 Formula CDM을 채점합니다. CDM에는 Node.js, ImageMagick, `pdflatex`가 필요합니다.
- <strong>`05_eval_kogovdoc.sh`</strong>는 KoGovDoc-Bench 294 페이지를 내려받아 모델 학습에 쓴 프롬프트로 파싱하고
  참조 Markdown과 비교해 Text NED를 채점합니다. 이어서 두 가지 프로토콜로 검색과 청크
  품질을 채점합니다:
  - **v1 프로토콜** (README "주요 결과"): Qwen2.5-1.5B-Instruct perplexity 모델로 BC/CS를 계산하고 semantic 청크 위에서
    BGE-M3로 Hit@K / MRR / nDCG를 계산합니다. 엔드포인트가 두 개 더 필요하며 스크립트에 나와 있습니다.
  - **RCPS 프로토콜** (EMNLP 2026 논문): Q–A 쌍 663개, 검색기 세 개, k는 {1, 5, 10}, 그리고 Qwen3-VL-2B-Instruct 기준
    Boundary Clarity. 논문은 공개 모델에 대해 RCPS 0.583, Hit@1 0.549, BC 0.610을 보고합니다.

## 6. 다른 문서에 맞게 바꾸기

- **프롬프트.** `training/prompts/templates.py`의 `VLM_TRAINING_SYSTEM_PROMPT`는 한국 정부 문서를
  설명합니다. 3단계 전에 자신의 도메인에 맞게 다시 쓰고 학습과 추론에 같은 프롬프트를 유지합니다.
- **Judge 샘플링.** `--sample-ratio`로 judge 시간과 coverage를 맞바꿉니다. 샘플링되지 않은 페이지는 채점 없이 학습에 들어갑니다.
- **불균형.** 몇몇 문서가 페이지 수의 대부분을 차지하면 `--max-doc-ratio`를 낮춥니다.
- **평가.** 파싱 벤치마크만이 아니라 직접 가진 문서에서 검색 성능으로 후보를 평가합니다. RCPS
  논문에서는 내재 경계 점수가 가장 좋은 파서가 검색에서 가장 나빴습니다
  ([wigtn/WigtnOCR-RADP](https://github.com/wigtn/WigtnOCR-RADP) 참고).

## 명령 확인 방법

전체 파이프라인을 v1 하드웨어에서 처음부터 끝까지 마지막으로 실행한 것은 2026년 3월입니다. 이번 공개에서는 그 하드웨어 없이
v1 실행이 저장해 둔 출력으로 각 명령을 확인했습니다:

| 명령 | 확인 |
|---|---|
| `python -m training.lora_trainer ... --dry-run` | 출력된 `swift sft` 인자가 모든 하이퍼파라미터에서 v1 `args.json`과 일치합니다 |
| `prepare_training_data.py --dry-run` | v1 arXiv pseudo-label과 judge 보고서로 실행하면 v1 개수인 646 페이지가 나옵니다 |
| `swift sft`, `swift export`, `vllm serve` 플래그 | 각 플래그가 ms-swift 4.0.1과 vLLM 0.13.0에 있습니다 |
| `run_eval.py` (OmniDocBench) | 공개 모델의 저장된 출력 1,276개와 v1.5 정답 데이터로 실행: Text NED 0.2884, Table TEDS 0.6490, Reading Order 0.2111, 건너뛴 페이지 79개로 v1 결과 파일과 일치합니다 |
| `run_val_eval.py --eval-only` (KoGovDoc) | 294 페이지 중 289 페이지에서 Text NED 0.2852로 v1 결과 파일과 일치합니다 |
| `run_retrieval_eval.py` (KoGovDoc v1) | 저장된 WigtnOCR-2B 출력에서 같은 질의 564개를 다시 만듭니다. 다른 BGE-M3 서버(`serve_bge_m3.py`)를 쓰면 semantic 청크 경계가 움직이므로 Hit@K와 MRR은 v1 표와 정확히 일치하지 않습니다 |
| `run_rcps_eval.py` | CPU에서 BGE-M3: Hit@1 0.5445, MRR@10 0.5928, nDCG@10 0.6152로 RCPS 실행과 같습니다 |
| `compute_parser_bc.py` (Boundary Clarity) | Apple-silicon GPU(`--device mps`)에서 실행: 경계 1,161개, 평균 BC 0.6100, 논문 보고값은 0.610입니다 |

이번 공개를 위해 다시 실행하지 않은 것: GPU 단계(`vllm serve`, `swift sft`, `swift export`, 모델 추론),
Formula CDM, 그리고 Qwen2.5-1.5B-Instruct perplexity 서버가 필요한 v1 BC/CS 스크립트.
