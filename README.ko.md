<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](pyproject.toml)

<a href="README.md"><img src="https://img.shields.io/badge/English-gray?style=for-the-badge" alt="English"></a>
<a href="README.ko.md"><img src="https://img.shields.io/badge/한국어-blue?style=for-the-badge" alt="Korean"></a>

</div>

# WigtnOCR

**VLM 기반 한국 공공문서 지능형 파서**

> Vision-Language Model과 LoRA 파인튜닝을 활용한 한국 공공기관 문서의 구조 보존 파싱 연구

WigtnOCR은 3계층 연구 프레임워크입니다:

1. **Framework** (`wigtnocr/`) — 설치 가능한 Python 패키지 (`pip install wigtnocr`), 하이브리드 라우팅 파서
2. **Fine-tuned Model** (`training/`) — 한국 공공문서 특화 Qwen3-VL-2B LoRA 어댑테이션
3. **Benchmark** (`evaluation/`) — KoGovDoc-Bench + OmniDocBench 평가 체계

---

## 빠른 시작

```bash
# 설치
pip install -e .

# 문서 파싱 (3줄)
from wigtnocr import WigtnOCR

ocr = WigtnOCR(mode="hybrid")
result = ocr.parse("document.pdf")
print(result.content)
```

---

## 아키텍처

### 하이브리드 라우팅 파이프라인

WigtnOCR은 문서 복잡도 분석을 통해 자동으로 파서를 선택합니다:

```
PDF 입력
    |
    v
[복잡도 분석]
  스캔 여부 / 표 개수 / 이미지 개수 / 텍스트 밀도
    |
    +-- 단순 (score < 0.5) --> CPU: PyMuPDF 또는 RapidOCR
    |
    +-- 복잡 (score >= 0.5) --> GPU: Two-Stage (추출 + VLM 구조화)
```

### 4-Parser 설계

| 파서 | Stage 1 (추출) | Stage 2 (구조화) | 용도 |
|------|---------------|-----------------|------|
| **Text-Baseline** | PyMuPDF | - | 디지털 PDF, 속도 우선 |
| **Image-Baseline** | RapidOCR | - | 스캔 PDF, 속도 우선 |
| **Text-Advanced** | PyMuPDF | Qwen3-VL-2B | 디지털 PDF, 구조 우선 |
| **Image-Advanced** | RapidOCR | Qwen3-VL-2B | 스캔 PDF, 구조 우선 |

### VLM 서빙

듀얼 RTX PRO 6000 (각 96GB)에서 vLLM으로 3단계 모델 서빙:

| 모델 | 포트 | 용도 |
|------|------|------|
| Qwen3-VL-30B-A3B-Thinking | 8000 | Pseudo GT 생성 |
| Qwen3-VL-8B-Thinking-FP8 | 8004 | 검증 / Ablation |
| Qwen3-VL-2B-Instruct | 8010 | 프로덕션 추론 + LoRA 베이스 |

---

## 3계층 기여 구조

### Layer 1: Framework (`wigtnocr/`)

설치 가능한 Python 패키지:
- **하이브리드 라우팅** — 복잡도 기반 자동 파서 선택
- **Two-stage 파이프라인** — 텍스트 추출 + VLM 구조화
- **시맨틱 청킹** — LangChain SemanticChunker + BGE-M3 임베딩
- **MoC 기반 메트릭** — BC (Boundary Clarity), CS (Chunk Stickiness)

```python
from wigtnocr import WigtnOCR, TwoStageParser, HybridParser

# 상위 API
ocr = WigtnOCR(mode="hybrid")
result = ocr.parse("document.pdf")

# 하위 API
parser = TwoStageParser(
    structurer_api_url="http://localhost:8010/v1/chat/completions",
    structurer_model="qwen3-vl-2b-instruct",
)
result = parser.parse_auto(pdf_bytes)
```

### Layer 2: Fine-tuned Model (`training/`)

Qwen3-VL-2B LoRA 파인튜닝 파이프라인:

| 파라미터 | 값 |
|---------|-----|
| 베이스 모델 | Qwen3-VL-2B-Instruct |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| 타겟 모듈 | q_proj, v_proj, k_proj, o_proj |
| Dropout | 0.05 |

**Pseudo GT 생성**: Claude/GPT-4o가 문서에서 구조화된 마크다운을 생성하며, CER < 5%일 때 자동 승인됩니다.

### Layer 3: Benchmark (`evaluation/`)

| 벤치마크 | 언어 | 문서 수 | 목적 |
|---------|------|---------|------|
| **KoGovDoc-Bench** | 한국어 | 공공기관 문서 11개 | 도메인 특화 평가 |
| **OmniDocBench** | 다국어 | 다수 | 범용 일반화 평가 |
| **DocVQA** | 영어 | 다수 | QA 기반 평가 (예정) |

**평가 지표**: CER, WER, Structure F1, TEDS, BC, CS

---

## 설치

### 사전 요구사항

- Python 3.11+
- CUDA 지원 GPU (VLM 추론용)

### 설치 방법

```bash
# 코어만
pip install -e .

# OCR 지원 (RapidOCR)
pip install -e ".[ocr]"

# 학습 의존성
pip install -e ".[training]"

# 평가 의존성
pip install -e ".[evaluation]"

# 전체
pip install -e ".[all]"
```

### VLM 서버

```bash
# vLLM 사용 (전체 설정은 docker-compose.vlm.yml 참고)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --served-model-name qwen3-vl-2b-instruct \
    --port 8010
```

---

## 사용법

### 문서 파싱

```bash
# 전체 데이터셋 파서 평가
python -m evaluation.cli parse --all

# 단일 문서
python -m evaluation.cli parse --input datasets/documents/kogov_001/doc.pdf
```

### 청킹 평가

```bash
# 청킹 품질 평가
python -m evaluation.cli chunk --all

# Mock 모드 (임베딩 API 없이)
python -m evaluation.cli chunk --all --use-mock
```

### 학습

```bash
# Pseudo Ground Truth 생성
python scripts/generate_pseudo_gt.py

# LoRA 파인튜닝 실행
python -m training.lora_trainer --config configs/training.yaml
```

---

## 평가 지표

| 단계 | 지표 | 공식 | 역할 |
|------|------|------|------|
| **전제 검증** | CER | `(S+D+I)/N` | 텍스트 추출 품질 |
| **전제 검증** | WER | `(S+D+I)/N` | 단어 수준 오류율 |
| **핵심** | Structure F1 | `2*P*R/(P+R)` | 구조 보존도 |
| **핵심** | TEDS | Tree Edit Distance Similarity | 표 구조 정확도 |
| **다운스트림** | BC | `1 - cosine_sim(chunk_i, chunk_i+1)` | 경계 독립성 |
| **다운스트림** | CS | Structural Entropy | 청크 의존도 |

---

## 프로젝트 구조

```
wigtnocr/                          # PyPI 패키지 (pip install wigtnocr)
    __init__.py                    # WigtnOCR 메인 클래스 (3줄 API)
    parsers/
        base.py                    # 추상 BaseParser, ParseResult
        pymupdf.py                 # 텍스트 추출 (PyMuPDF)
        rapidocr.py                # 이미지 OCR (RapidOCR)
        vlm.py                     # VLM 구조화 (TextStructurer)
    pipeline/
        two_stage.py               # Two-stage 파서 (추출 + VLM)
        hybrid.py                  # 하이브리드 라우터 (복잡도 기반)
    chunking/
        chunker.py                 # SemanticChunker (LangChain)
        embeddings.py              # API 임베딩 클라이언트 (BGE-M3)
        metrics.py                 # BC/CS 메트릭
    utils/
        markdown.py                # 마크다운 유틸리티

training/                          # LoRA 파인튜닝 파이프라인
    config.py                      # LoRA/Training 데이터클래스
    gt_generator.py                # Pseudo GT 생성 (Claude/GPT-4o)
    data_prep.py                   # 학습 데이터 준비
    lora_trainer.py                # LoRA 파인튜닝 진입점
    prompts/
        templates.py               # GT/학습용 시스템 프롬프트

evaluation/                        # 벤치마크 및 평가 체계
    metrics/
        cer.py                     # CER, WER (jiwer)
        structure.py               # Structure F1
        teds.py                    # Table Edit Distance Similarity
        chunking.py                # BC/CS 래퍼
    benchmarks/
        kogovdoc.py                # KoGovDoc-Bench 로더
        omnidocbench.py            # OmniDocBench 어댑터
        docvqa.py                  # DocVQA (예정)
    runners/
        parser_eval.py             # 파서 평가 러너
        chunking_eval.py           # 청킹 평가 러너
    cli.py                         # 통합 CLI (parse/chunk)

configs/                           # YAML 설정
    default.yaml                   # VLM 서버 엔드포인트
    training.yaml                  # LoRA 하이퍼파라미터
    evaluation.yaml                # 평가 설정

datasets/                          # 벤치마크 데이터셋
    papers/                        # 영어 학술 논문 (39편)
    documents/                     # 한국 공공문서 (11건)
    omnidocbench/                  # OmniDocBench

scripts/                           # 유틸리티 스크립트
    generate_pseudo_gt.py          # Pseudo GT 생성 스크립트
    setup_datasets.py              # 데이터셋 셋업 헬퍼
```

---

## 참고 문헌

- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) — 베이스 VLM
- [MoC 논문](https://arxiv.org/abs/2503.09600) — BC/CS 청킹 메트릭
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench) — 문서 벤치마크
- [vLLM](https://github.com/vllm-project/vllm) — VLM 서빙 엔진

---

## 라이선스

MIT License

## 인용

```bibtex
@software{wigtnocr,
  title = {WigtnOCR: VLM-based Korean Government Document Intelligence Parser},
  author = {Kim, Hyeongseob},
  year = {2026},
  url = {https://github.com/Hyeongseob91/wigtnocr}
}
```
