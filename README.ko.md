<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](pyproject.toml)

<a href="README.md"><img src="https://img.shields.io/badge/English-gray?style=for-the-badge" alt="English"></a>
<a href="README.ko.md"><img src="https://img.shields.io/badge/한국어-blue?style=for-the-badge" alt="Korean"></a>

</div>

# WigtnOCR

**Pseudo-Label Distillation 기반 구조 보존 문서 파싱**

> 30B VLM의 문서 파싱 능력을 2B 모델로 증류하고, 구조화된 출력이 RAG 청킹과 검색 성능을 개선함을 증명합니다.

WigtnOCR은 Qwen3-VL-30B(teacher)의 문서 파싱 능력을 품질 필터링된 pseudo-labeling과 LoRA fine-tuning으로 Qwen3-VL-2B(student)에 증류하는 연구 프레임워크입니다. OmniDocBench(국제 벤치마크)와 KoGovDoc(한국 공공문서)에서 평가하며, 2단계 인과 평가로 **구조화 파싱 → 청킹 품질 → 검색 성능** 인과 관계를 증명합니다.

---

## 주요 결과

### OmniDocBench — 증류 효과 (RQ1-3)

| 지표 | 2B Base | **WigtnOCR-2B** | 30B Teacher | Marker | 방향 |
|------|:-------:|:---------------:|:-----------:|:------:|:----:|
| Text NED | 0.364 | **0.288** | 0.289 | 0.218 | 낮을수록 좋음 |
| Table TEDS | 0.561 | **0.649** | 0.523 | 0.586 | 높을수록 좋음 |
| Formula CDM F1 | 0.865 | **0.884** | 0.939 | 0.863 | 높을수록 좋음 |
| Reading Order | 0.300 | **0.211** | 0.227 | 0.165 | 낮을수록 좋음 |
| Skip Rate | 18.8% | **5.8%** | 5.5% | 0.4% | 낮을수록 좋음 |

**Student가 Teacher를 5개 중 4개 카테고리에서 매칭 또는 초과.** Table TEDS는 Teacher 대비 +12.6pp 우위.

### KoGovDoc — 청킹 품질 (RQ4, 진행 중)

MoC 프레임워크(Zhao et al., ACL 2025) 기반 BC/CS 평가. Qwen2.5-1.5B로 perplexity 계산.

| 파서 | 전략 | BC ↑ | CS ↓ |
|------|------|:----:|:----:|
| **WigtnOCR-2B** | Header-based | ___ | ___ |
| WigtnOCR-2B | Semantic | ___ | ___ |
| PaddleOCR | Semantic | ___ | ___ |
| PaddleOCR | Fixed-size | ___ | ___ |

### KoGovDoc — 검색 성능 (RQ5, 예정)

| 파서 | Hit@1 ↑ | Hit@5 ↑ | MRR ↑ | nDCG@10 ↑ |
|------|---------|---------|-------|-----------|
| WigtnOCR-2B | ___ | ___ | ___ | ___ |
| PaddleOCR | ___ | ___ | ___ | ___ |

---

## 작동 방식

### 4단계 파이프라인

```
Stage 1: Pseudo GT 생성
    PDF 페이지 → Qwen3-VL-30B (teacher) → 구조화된 마크다운
    4,501 페이지, 49개 문서 (한국어 + 영어)

Stage 2: 품질 검증
    GT 마크다운 → Qwen3.5-122B (judge, 텍스트 전용) → 점수 1-5
    74-75% 합격률, 5차원 품질 평가

Stage 3: 학습 데이터 준비
    품질 필터링 (점수 >= 3) + 편향 보정 + train/val 분할

Stage 4: LoRA 파인튜닝
    Qwen3-VL-2B + LoRA (rank=8, alpha=32) → WigtnOCR-2B
    ms-swift, DeepSpeed ZeRO-2, 3 에폭
```

### 2단계 인과 평가 (KoGovDoc)

```
Step 1: 구조화가 더 좋은 청크를 만드는가?
  WigtnOCR (구조화) vs PaddleOCR (비구조화)
  → 3가지 청킹 전략 → BC/CS (perplexity 기반, MoC)

Step 2: 더 좋은 청크가 더 좋은 검색을 만드는가?
  최적 청크 → BGE-M3 임베딩 → FAISS → Hit@K, MRR, nDCG
```

---

## 학습 데이터셋

| 데이터셋 | 문서 수 | 페이지 | 언어 | 출처 |
|---------|--------|-------|------|------|
| **KoGovDoc** | 10 | 3,637 | 한국어 | 정부 공공문서 |
| **ArXivPapers** | 39 | 864 | 영어 | arXiv (cs.CL, cs.CV, cs.LG) |
| **합계** | 49 | 4,501 | 이중 언어 | — |

GT는 Qwen3-VL-30B로 생성, Qwen3.5-122B로 검증 (텍스트 전용 judge, 5차원 품질 평가).

---

## 빠른 시작

```bash
# 설치
pip install -e ".[all]"

# 1. Pseudo GT 생성
python scripts/training/generate_pseudo_gt.py --dataset documents --batch-size 4

# 2. GT 품질 검증
python scripts/training/validate_gt.py --dataset documents --sample-ratio 0.3

# 3. 학습 데이터 준비
python scripts/training/prepare_training_data.py --min-score 3 --max-doc-ratio 0.25

# 4. LoRA 파인튜닝
python -m training.lora_trainer --config configs/training.yaml

# 5. OmniDocBench 평가
python scripts/eval_omnidocbench/run_inference.py --model wigtnocr-2b-v1 --output-dir results/omnidocbench/v1
python scripts/eval_omnidocbench/run_eval.py --pred-dir results/omnidocbench/v1

# 6. KoGovDoc 평가
python scripts/eval_kogovdoc/run_val_eval.py --model wigtnocr-2b-v1 --output-dir results/kogovdoc/v1_val
python scripts/eval_kogovdoc/run_paddleocr_baseline.py --output-dir results/kogovdoc/paddleocr_val
python scripts/eval_kogovdoc/run_bccs_eval.py --ppl-model qwen2.5-1.5b-instruct
```

---

## 모델 구성

| 역할 | 모델 | 용도 |
|------|------|------|
| Teacher | Qwen3-VL-30B-A3B-Instruct (FP8) | Pseudo GT 생성 |
| Judge | Qwen3.5-122B-A10B-NVFP4 | GT 품질 검증 (텍스트 전용) |
| Student | Qwen3-VL-2B-Instruct + LoRA r=8 | **배포 모델 (WigtnOCR-2B)** |
| BC/CS PPL | Qwen2.5-1.5B-Instruct | MoC perplexity 계산 |
| Embedding | BAAI/bge-m3 (Infinity) | Semantic chunking + retrieval |

모든 VLM은 vLLM v0.13.0으로 듀얼 RTX PRO 6000 (각 98GB), TP=2에서 서빙.

---

## 평가 지표

| 벤치마크 | 지표 | 방법 | 목적 |
|---------|------|------|------|
| OmniDocBench | Text NED | 편집 거리 | 텍스트 인식 품질 |
| OmniDocBench | Table TEDS | 트리 편집 거리 | 표 구조 정확도 |
| OmniDocBench | Formula CDM | 시각적 문자 매칭 | 수식 인식 |
| OmniDocBench | Reading Order NED | 순서 NED | 레이아웃 이해 |
| KoGovDoc | BC | ppl(q\|d) / ppl(q) | 청크 경계 독립성 |
| KoGovDoc | CS | Structural Entropy | 청크 의존도 (낮을수록 좋음) |
| KoGovDoc | Hit@K, MRR, nDCG | FAISS 검색 | 검색 성능 |

BC/CS는 MoC 프레임워크(Zhao et al., ACL 2025)에 따라 perplexity 기반으로 계산.

---

## 프로젝트 구조

```
wigtnocr/                          # 코어 라이브러리
    parsers/                       # PyMuPDF, RapidOCR, VLM 구조화기
    chunking/                      # 청커, 임베딩, BC/CS 메트릭
    pipeline/                      # Two-stage, 하이브리드 라우팅
    utils/                         # 마크다운 유틸리티

training/                          # Pseudo-labeling + LoRA 파이프라인
    gt_generator.py                # 30B teacher → 마크다운 GT
    gt_validator.py                # 122B judge → 품질 점수
    data_prep.py                   # 품질 필터링, 편향 보정
    lora_trainer.py                # ms-swift LoRA 파인튜닝

evaluation/                        # 평가 프레임워크
    metrics/                       # NED, TEDS, CER, BC/CS (perplexity), 검색
    omnidocbench/                  # OmniDocBench 평가기
    benchmarks/                    # 벤치마크 어댑터
    runners/                       # 평가 러너

scripts/
    training/                      # GT 생성, 검증, 데이터 준비
    eval_omnidocbench/             # OmniDocBench 추론 + 평가
    eval_kogovdoc/                 # KoGovDoc val 평가 + BC/CS + 검색
```

---

## 참고 문헌

- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) — 베이스 VLM
- [ms-swift](https://github.com/modelscope/ms-swift) — 공식 Qwen 파인튜닝 프레임워크
- [MoC](https://arxiv.org/abs/2503.09600) — BC/CS 청킹 품질 메트릭 (ACL 2025)
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench) — 문서 파싱 벤치마크 (CVPR 2025)
- [vLLM](https://github.com/vllm-project/vllm) — VLM 서빙 엔진
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — 다국어 임베딩 모델
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — OCR 베이스라인 (RapidOCR 경유)

---

## 라이선스

MIT License

## 인용

```bibtex
@software{wigtnocr2026,
  title   = {WigtnOCR: Pseudo-Label Distillation for Structure-Preserving Document Parsing},
  author  = {Kim, Hyeongseob},
  year    = {2026},
  url     = {https://github.com/Hyeongseob91/research-vlm-based-document-parsing}
}
```
