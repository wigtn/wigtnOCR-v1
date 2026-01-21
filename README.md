# VLM Document Parsing Quality Test & Tech Report

> **"Does Structural Integrity in Parsing Improve Semantic Retrieval?"**
>
> 본 프로젝트는 VLM(Qwen3-VL)의 구조화된 마크다운 출력이 전통적인 OCR 방식 대비 **Semantic Chunking** 및 **RAG(Retrieval-Augmented Generation)** 성능에 미치는 영향을 정량적으로 분석하는 프레임워크입니다.

---

## Quick Start: 언제 이걸 쓰나요?

- 복잡한 PDF를 RAG에 넣었는데, **답변이 자꾸 엉키는 경우**
- VLM 파서 vs 기존 OCR 파서를 **정량 비교**하고 싶은 경우
- 한국어/멀티컬럼/표가 많은 문서에 **최적 청킹 전략**을 찾고 싶은 경우
- 스캔된 PDF에서 **VLM만 텍스트 추출이 가능**한지 확인하고 싶은 경우

```bash
# 30초 만에 시작하기
uv sync
python -m src.test_parsers --pdf your_document.pdf --gt your_ground_truth.md
```

---

## 1. Research Motivation

기존 RAG 파이프라인은 주로 Plain Text에 의존해 왔습니다. 그러나 표(Table), 계층 구조(Headers), 목록(Lists)이 포함된 복잡한 문서에서 구조 정보의 손실은 곧 **의미적 단절(Semantic Discontinuity)**로 이어집니다.

### Problem Statement

- **구조 손실**: 전통적인 OCR은 레이아웃 순서가 꼬이거나 표 내부 데이터를 단순 텍스트로 치환하여 청킹(Chunking) 시 맥락을 파괴
- **Multi-column 오류**: 2단 레이아웃에서 좌→우가 아닌 상→하로 읽어 문장이 뒤섞임
- **표 구조 붕괴**: 행-열 관계가 파괴되어 데이터 의미 상실

### Hypothesis

> VLM 기반의 **Agentic Reasoning**을 통한 마크다운 파싱은 문서의 논리적 구조를 보존하며, 이는 Boundary Score와 Chunk Score를 개선하여 최종적인 검색 정확도를 높일 것이다.

---

## 2. Methodology

### 2.1 Multi-Modal Agentic Parsing

```
┌─────────────────────────────────────────────────────────────────┐
│  Document (PDF/Image)                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           PDF → Image Conversion (150 DPI)               │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Parsing Layer (3 Methods)                   │   │
│  │  ┌───────────────┬───────────────┬───────────────┐      │   │
│  │  │   VLM         │  pdfplumber   │   RapidOCR    │      │   │
│  │  │  (Qwen3-VL)   │   (Digital)   │   (Scanned)   │      │   │
│  │  │               │               │               │      │   │
│  │  │  Structured   │  Plain Text   │  Plain Text   │      │   │
│  │  │  Markdown     │  + Tables     │  (OCR only)   │      │   │
│  │  └───────────────┴───────────────┴───────────────┘      │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Evaluation Metrics                          │   │
│  │         CER • WER • Latency • BS • CS                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

Qwen3-VL-Thinking 모델은 이미지 내 요소를 단순히 읽는 것이 아니라, 내부 추론(`Chain-of-Thought`) 과정을 통해 요소 간 관계를 정의합니다.

- **Layout Understanding**: 문서의 전체 레이아웃 파악 및 읽기 순서 결정
- **Table Structure Recognition**: 표 내부의 행-열 관계 및 헤더 인식
- **Hierarchical Heading**: 제목-부제목-본문의 계층 구조 보존

### 2.2 Evaluation Metrics

#### Phase 1: Lexical Accuracy (어휘적 정확도)

문서 인식의 근본적인 정확도를 측정하기 위해 편집 거리(Levenshtein Distance) 기반 알고리즘을 사용합니다.

| Metric | Formula | Description |
|--------|---------|-------------|
| **CER** | `(S + D + I) / N` | 문자 단위 오류율. N = GT 문자 수 |
| **WER** | `(S + D + I) / N` | 단어 단위 오류율. N = GT 토큰 수 |

- `S`: Substitution (교체) - 잘못 인식된 문자/단어
- `D`: Deletion (삭제) - 누락된 문자/단어
- `I`: Insertion (삽입) - 추가된 문자/단어 (hallucination 포함)

**한국어 특화**: 교착어 특성상 공백 기반 토큰화가 부적절하므로 `MeCab/Okt` 형태소 분석기를 통한 토큰화 후 WER 측정을 병행합니다.

#### Phase 2: Semantic Integrity (의미적 완전성)

구조화된 데이터가 청킹 품질에 미치는 영향을 측정합니다.

| Metric | Description |
|--------|-------------|
| **Boundary Score (BS)** | 의미적으로 분리되어야 할 지점에서 정확히 청크가 나뉘었는가? |
| **Chunk Score (CS)** | 나뉜 청크가 독립적으로 완전한 의미 정보를 포함하는가? |

### 2.3 How to Read the Metrics

| Metric | Good | Bad | Interpretation |
|--------|------|-----|----------------|
| **CER** | < 10% | > 30% | 0에 가까울수록 문자 인식 정확 |
| **WER** | < 15% | > 40% | 0에 가까울수록 단어 인식 정확 |
| **BS** | > 0.8 | < 0.5 | 1에 가까울수록 경계 분리 정확 |
| **CS** | > 0.8 | < 0.5 | 1에 가까울수록 청크 품질 우수 |

**핵심 인사이트**: VLM 파서는 CER/WER이 약간 나빠도 BS/CS가 높아, RAG 검색 정확도에서 이득을 줄 수 있습니다. 단순 텍스트 정확도보다 **구조 보존**이 더 중요한 경우가 많습니다.

---

## 3. Technical Specifications

| Component | Specification |
|-----------|---------------|
| **VLM Model** | `Qwen3-VL-8B-Thinking` (OpenAI Vision API Compatible) |
| **VLM Server** | `vLLM` with CUDA Acceleration |
| **OCR Engine (Digital)** | `pdfplumber` - 디지털 PDF 텍스트 추출 |
| **OCR Engine (Scanned)** | `RapidOCR` via Docling - 이미지 기반 OCR |
| **PDF to Image** | `pdf2image` (poppler) / `pdfplumber` fallback |
| **Tokenizer (Korean)** | `KoNLPy` - MeCab (권장) / Okt (순수 Python) |
| **Tokenizer (English)** | Whitespace split |
| **Evaluation Library** | `jiwer` - CER/WER 계산 |

---

## 4. Experimental Design

### 4.1 Test Datasets

| ID | File | Type | Pages | GT Source |
|----|------|------|-------|-----------|
| test_1 | `test_data_1.pdf` | Digital PDF (Korean) | 4 | Manual Transcription |
| test_2 | `test_data_2.jpg` | Image | 1 | Manual Transcription |
| test_3 | `Chain-of-Thought...pdf` | Scanned PDF (Paper) | 4 | Manual Transcription |
| test_4 | `2025_한국부자보고서.pdf` | Digital PDF (Korean) | - | Manual Transcription |

### 4.2 Parser Comparison

| Parser | Input | Output Format | Pros | Cons |
|--------|-------|---------------|------|------|
| **VLM** | Image | Structured Markdown | 구조 보존, 스캔 지원 | 느림, GPU 필요 |
| **pdfplumber** | PDF | Plain Text + Tables | 빠름, 정확함 | 디지털 전용, 구조 손실 |
| **RapidOCR** | Image | Plain Text | 스캔 지원 | 구조 손실, 한국어 약함 |

### 4.3 Ground Truth 작성 원칙

CER/WER 평가를 위한 GT는 **원문 전사(Verbatim Transcription)** 방식으로 작성합니다.

```
┌─────────────────────────────────────────────────────────────┐
│  Ground Truth 작성 원칙                                     │
├─────────────────────────────────────────────────────────────┤
│  1. 원문 그대로 전사 (요약/정리 X)                          │
│  2. 페이지 단위 작성 권장 (전체보다 대표 페이지 선정)       │
│  3. 읽기 순서대로 텍스트 나열                               │
│  4. 마크다운으로 구조 보존 (헤더, 표, 리스트)               │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Ground Truth Examples

실제 GT 파일을 참고하여 동일한 스타일로 작성하세요:

- `data/test_1/gt_data_1.md`: 한국어 공고문 (표 포함)
- `data/test_3/gt_data_3.md`: 영어 논문 (섹션 구조)

---

## 5. Prompt Engineering (Tech Report)

### 5.1 Problem: VLM Hallucination

초기 프롬프트 사용 시 VLM이 **요약, 설명, 추론**을 추가하여 `Insertion` 에러가 급증하는 문제가 발생했습니다.

**기존 프롬프트 (v1)**:
```
You are a document structure extraction expert.
Convert this document image to well-structured Markdown.
...
```

**문제점**:
- VLM이 "extraction expert"로서 내용을 **해석하고 정리**하려는 경향
- 원문에 없는 설명 문장 추가 → `I` (Insertion) 에러 증가
- CER 수치가 비정상적으로 높아짐

### 5.2 Solution: Transcription-focused Prompt

**개선된 프롬프트 (v2)**:
```
You are a document TRANSCRIPTION engine, not a writer.
Your job is to CONVERT the given document image into Markdown by
STRICTLY TRANSCRIBING what is visible in the image.

## Hard Constraints (must follow):
- DO NOT add, rephrase, summarize, infer, or translate any text.
- DO NOT explain, comment, or describe what you are doing.
- If something is partially cut off or unreadable, write `[UNREADABLE]` instead of guessing.
- If a value is missing in the image, leave it blank or use `[EMPTY]`. Never invent values.

## Markdown Formatting Rules:
1. Headers: use `#` for the main title, `##` for sections, `###` for subsections
2. Tables: preserve rows/columns as Markdown tables using `|` and `---`
3. Lists: use `-` for bullets, `1.` for numbered lists
4. Images/Charts: if there is a visible caption, transcribe it; otherwise use `[Figure]`
5. Forms: use `Field: Value` exactly as written

## Important:
- Follow the reading order a human would use (left to right, top to bottom).
- Preserve line breaks and spacing where they matter for meaning (e.g., in tables or forms).
- Output VALID Markdown ONLY. No extra text before or after.

## Output:
(Write ONLY the transcribed Markdown content here.)
```

### 5.3 Key Changes

| Aspect | v1 (기존) | v2 (개선) |
|--------|----------|----------|
| **Role** | "extraction expert" | "TRANSCRIPTION engine" |
| **Goal** | 구조 추출 | 원문 전사 |
| **Constraints** | 암시적 | 명시적 Hard Constraints |
| **Hallucination** | 허용 | `[UNREADABLE]`, `[EMPTY]` 사용 |
| **Output** | Markdown + 설명 가능 | Markdown ONLY |

### 5.4 Expected Improvement

| Metric | Before (v1) | After (v2) | Change |
|--------|-------------|------------|--------|
| **CER** | ~56% | TBD | ↓ Expected |
| **Insertion (I)** | 2,559 | TBD | ↓↓ Expected |
| **Hallucination** | High | Low | ↓↓ Expected |

---

## 6. Preliminary Results

### 6.1 Observations

| Insight | Description |
|---------|-------------|
| **VLM 강점** | 스캔 PDF에서 pdfplumber/RapidOCR 완전 실패, VLM만 텍스트 추출 성공 |
| **Latency Trade-off** | VLM은 전통적 방식 대비 3~10배 느림 → Hybrid 전략 필요 |
| **Hallucination** | Thinking 모델 사용 시 설명이 추가되어 Insertion 에러 발생 경향 |

### 6.2 Example: Scanned PDF (test_3)

| Parser | CER | WER | Latency | Note |
|--------|-----|-----|---------|------|
| **VLM** | 56.29% | 70.85% | 15.61s | 유일하게 텍스트 추출 성공 |
| pdfplumber | 99.62% | 100% | 18.12s | 스캔 PDF → 추출 실패 |
| RapidOCR | N/A | N/A | 6.85s | 완전 실패 (0 chars) |

**핵심 발견**: 스캔된 PDF에서는 VLM이 **유일한 선택지**입니다.

### 6.3 Hybrid Strategy Proposal

```python
def select_parser(pdf_bytes):
    pdf_type = detect_pdf_type(pdf_bytes)

    if pdf_type == "digital" and has_text:
        return "pdfplumber"  # Fast & Accurate
    elif pdf_type == "scanned" or has_complex_layout:
        return "VLM"         # Structure Preservation
    else:
        return "RapidOCR"    # Fallback
```

---

## 7. Known Issues & Error Patterns

### 7.1 VLM Hallucination

**증상**: VLM이 원문에 없는 설명/요약을 추가하여 Insertion 에러 급증

**Mitigation**:
- Transcription-focused 프롬프트 사용 (섹션 5.2 참조)
- `max_tokens` 제한으로 과도한 생성 방지
- `temperature: 0.1` 로 창의성 억제

### 7.2 RapidOCR 한국어 인식 약함

**증상**: 한국어 문서에서 인식률이 현저히 낮음

**Mitigation**:
- 스캔 품질 개선 (DPI 증가)
- 한국어 문서는 VLM 또는 pdfplumber 우선 사용
- 후처리 규칙 추가 (특수문자 정리)

### 7.3 Multi-column 레이아웃 순서 꼬임

**증상**: 2단 레이아웃에서 텍스트 순서가 뒤섞임

**Mitigation**:
- VLM 우선 사용 (레이아웃 이해 능력)
- Column-wise crop 후 개별 OCR
- 후처리로 순서 재정렬

### 7.4 GT와 Parser 출력 범위 불일치

**증상**: GT는 1페이지, Parser는 전체 페이지 추출 → CER 2000%+

**Mitigation**:
- GT와 동일한 페이지 범위로 파싱 제한
- GT 작성 시 전사 범위 명시
- 페이지 단위로 개별 평가

---

## 8. Project Structure

```
test-vlm-document-parsing/
├── src/
│   ├── app.py                 # Streamlit 비교 UI
│   ├── test_parsers.py        # CLI 벤치마크 스크립트
│   └── parsers/
│       ├── __init__.py
│       ├── vlm_parser.py      # VLM Parser (Qwen3-VL)
│       ├── ocr_parser.py      # pdfplumber + ImageOCR
│       └── docling_parser.py  # Docling (RapidOCR)
├── data/
│   ├── test_1/                # 한국어 공고문 PDF + GT
│   ├── test_2/                # Image + GT
│   ├── test_3/                # 영어 논문 (스캔) + GT
│   └── test_4/                # 한국어 보고서 + GT
├── pyproject.toml
└── README.md
```

---

## 9. How to Run

### 9.1 Installation

```bash
# uv 사용 (권장)
uv sync

# 또는 pip
pip install -e .

# 한국어 평가를 위한 MeCab 설치 (Ubuntu)
sudo apt-get install mecab mecab-ko mecab-ko-dic libmecab-dev
pip install mecab-python3
```

### 9.2 Running Benchmark

```bash
# 기본 실행 (3개 파서 비교)
python -m src.test_parsers \
    --pdf data/test_1/test_data_1.pdf \
    --gt data/test_1/gt_data_1.md

# 한국어 토크나이저 사용
python -m src.test_parsers \
    --pdf data/test_4/2025_한국부자보고서.pdf \
    --gt data/test_4/gt_data_4.md \
    --tokenizer mecab

# 결과 저장
python -m src.test_parsers \
    --pdf data/test_3/test_data_3.pdf \
    --gt data/test_3/gt_data_3.md \
    --output-dir results/test_3

# 옵션
--skip-vlm       # VLM 테스트 스킵 (GPU 없을 때)
--skip-docling   # Docling 테스트 스킵
--verbose        # 추출 결과 미리보기
--save-docs      # docs/Parsing_test_<날짜>/ 에 저장
```

### 9.3 Streamlit UI

```bash
cd src
streamlit run app.py --server.port 8501
```

---

## 10. Future Work

- [ ] **Prompt v2 검증**: Transcription-focused 프롬프트 효과 정량 평가
- [ ] **Semantic Chunking Integration**: 파싱된 마크다운 기반 `Hierarchical Chunking` 알고리즘 구현
- [ ] **Vector Search Benchmarking**: 파싱 결과물이 Vector DB 검색 Hit Rate에 미치는 영향 측정
- [ ] **LLM-based Evaluation**: GPT-4o 기반 `G-Eval`로 파싱 품질 정성 평가
- [ ] **Multi-page GT Automation**: VLM을 활용한 반자동 GT 생성 파이프라인

---

## 11. References

- [jiwer](https://github.com/jitsi/jiwer) - CER/WER calculation
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF text extraction
- [Docling](https://github.com/DS4SD/docling) - Document understanding with RapidOCR
- [KoNLPy](https://konlpy.org/) - Korean NLP toolkit
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) - Vision-Language Model

---

## License

MIT License
