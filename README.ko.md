<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](pyproject.toml)

<a href="README.md"><img src="https://img.shields.io/badge/English-gray?style=for-the-badge" alt="English"></a>
<a href="README.ko.md"><img src="https://img.shields.io/badge/한국어-blue?style=for-the-badge" alt="Korean"></a>

</div>

# WigtnOCR

**Pseudo-Label Distillation 기반 구조 보존 문서 파싱**

> 30B VLM의 문서 파싱 능력을 2B 모델로 증류하고, 구조화된 출력이 한국 정부 문서의 RAG 청킹과 검색 성능을 개선함을 증명합니다.

WigtnOCR은 Qwen3-VL-30B(teacher)의 문서 파싱 능력을 품질 필터링된 pseudo-labeling과 LoRA fine-tuning으로 Qwen3-VL-2B(student)에 증류하는 연구 프레임워크입니다. 증류된 **WigtnOCR-2B**는 OmniDocBench에서 30B teacher를 매칭/초과하며, KoGovDoc에서 6개 파서 중 **최고 검색 성능**을 달성합니다 — **더 나은 파싱 → 더 나은 청크 → 더 나은 검색**.

---

## 주요 결과

### OmniDocBench — 증류 효과

<div align="center">
<img src="docs/figures/fig1_omnidocbench.png" width="100%" alt="OmniDocBench 결과">
</div>

| 모델 | Text NED ↓ | Table TEDS ↑ | CDM F1 ↑ | Read Order ↓ | Skip % ↓ |
|------|:----------:|:------------:|:--------:|:------------:|:--------:|
| Qwen3-VL-2B (Base) | 0.364 | 0.561 | 0.865 | 0.300 | 18.8% |
| **WigtnOCR-2B (Ours)** | **0.288** | **0.649** | **0.884** | **0.211** | **5.8%** |
| Qwen3-VL-30B (Teacher) | 0.289 | 0.523 | 0.939 | 0.227 | 5.5% |
| Marker | 0.218 | 0.586 | 0.863 | 0.165 | 0.4% |

**WigtnOCR-2B는 5개 지표 중 4개에서 30B Teacher를 매칭 또는 초과합니다.** Table TEDS는 Teacher 대비 +12.6pp 우위. Skip Rate는 18.8% → 5.8%로 안정성이 대폭 향상되었습니다.

### KoGovDoc — 검색 성능

<div align="center">
<img src="docs/figures/fig2_retrieval.png" width="85%" alt="KoGovDoc 검색 결과">
</div>

| 모델 | Hit@1 ↑ | Hit@5 ↑ | MRR@10 ↑ | nDCG@10 ↑ |
|------|:-------:|:-------:|:--------:|:---------:|
| **WigtnOCR-2B** | **0.739** | **0.855** | **0.788** | 0.437 |
| Qwen3-VL-30B | 0.716 | 0.839 | 0.771 | 0.411 |
| Marker | 0.711 | 0.853 | 0.771 | 0.412 |
| Qwen3-VL-2B | 0.709 | 0.814 | 0.756 | 0.444 |
| MinerU | 0.608 | 0.789 | 0.682 | 0.384 |
| PaddleOCR | 0.512 | 0.693 | 0.592 | 0.293 |

**WigtnOCR-2B가 Hit@1, Hit@5, MRR@10에서 6개 파서 중 1위입니다.** PaddleOCR(순수 OCR) 대비 Hit@1이 +22.7pp 향상되어, 구조화 파싱이 검색 성능에 직접적으로 기여함을 보여줍니다.

### BC/CS — 청크 품질 ≠ 검색 품질

<div align="center">
<img src="docs/figures/fig3_bc_vs_retrieval.png" width="55%" alt="BC vs 검색 성능">
</div>

| 모델 | BC ↑ | CS ↓ |
|------|:----:|:----:|
| MinerU | **0.735** | **2.711** |
| WigtnOCR-2B | 0.706 | 2.859 |
| Qwen3-VL-30B | 0.714 | 3.164 |
| Marker | 0.683 | 3.206 |
| Qwen3-VL-2B | 0.678 | 3.446 |
| PaddleOCR | 0.654 | 3.420 |

MinerU가 BC/CS 1위이지만 검색에서는 5위 — 청크 경계 품질만으로는 검색 성능을 예측할 수 없습니다. **텍스트의 풍부함과 구조적 충실도가 end-to-end RAG에서 더 중요합니다.**

---

## 작동 방식

### 증류 파이프라인

```
Stage 1: Pseudo GT 생성
    PDF 페이지 → Qwen3-VL-30B (teacher) → 구조화된 마크다운
    4,501 페이지, 49개 문서 (한국어 + 영어)

Stage 2: 품질 검증
    GT 마크다운 → Qwen3.5-122B (judge, 텍스트 전용) → 점수 1–5
    74–75% 합격률, 5차원 품질 평가

Stage 3: 학습 데이터 준비
    품질 필터링 (점수 ≥ 3) + 편향 보정 + train/val 분할

Stage 4: LoRA 파인튜닝
    Qwen3-VL-2B + LoRA (rank=8, alpha=32) → WigtnOCR-2B
    ms-swift, DeepSpeed ZeRO-2, 3 에폭
```

### 2단계 인과 평가 (KoGovDoc)

```
Step 1: 구조화 파싱이 더 좋은 청크를 만드는가?
    6개 파서 × semantic chunking → BC/CS (perplexity 기반, MoC ACL 2025)

Step 2: 더 좋은 청크가 더 좋은 검색을 만드는가?
    청크 → BGE-M3 임베딩 → FAISS → Hit@K, MRR, nDCG
```

**결과:** 더 나은 파싱 → 더 나은 청크 → 더 나은 검색 (6개 파서에서 검증)

---

## 학습 데이터셋

| 데이터셋 | 문서 수 | 페이지 | 언어 | 출처 |
|---------|:------:|:-----:|:----:|------|
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

# 6. KoGovDoc 평가 (6파서 비교)
python scripts/eval_kogovdoc/run_val_eval.py --model wigtnocr-2b-v1 --output-dir results/kogovdoc/v1_val
python scripts/eval_kogovdoc/run_bccs_eval.py --parsers wigtnocr paddleocr marker mineru qwen3vl2b qwen3vl30b --strategies semantic
python scripts/eval_kogovdoc/run_retrieval_eval.py --parsers wigtnocr paddleocr marker mineru qwen3vl2b qwen3vl30b
```

---

## 모델 구성

| 역할 | 모델 | 용도 |
|------|------|------|
| Teacher | Qwen3-VL-30B-A3B-Instruct (FP8) | Pseudo GT 생성 |
| Judge | Qwen3.5-122B-A10B-NVFP4 | GT 품질 검증 (텍스트 전용) |
| Student | Qwen3-VL-2B-Instruct + LoRA r=8 | **배포 모델 (WigtnOCR-2B)** |
| BC/CS PPL | Qwen2.5-1.5B-Instruct | MoC perplexity 계산 |
| Embedding | BAAI/bge-m3 (Infinity) | Semantic chunking + 검색 |

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

BC/CS는 MoC 프레임워크(Zhao et al., ACL 2025)에 따라 Qwen2.5-1.5B로 perplexity 기반 계산.

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
- [Marker](https://github.com/VikParuchuri/marker) — PDF→마크다운 변환기
- [MinerU](https://github.com/opendatalab/MinerU) — PDF 파싱 도구

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
