# 3. Methodology

## 3.1 Evaluation Framework Overview

Our evaluation framework measures document parsing quality across three dimensions, each corresponding to a research question:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │   RQ1       │   │   RQ2       │   │   RQ3       │          │
│   │ Prerequisite│ → │  Structure  │ → │ Downstream  │          │
│   │ Validation  │   │ Preservation│   │   Impact    │          │
│   └─────────────┘   └─────────────┘   └─────────────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐             │
│   │ CER, WER  │    │Structure  │    │  BC, CS   │             │
│   │(전제 확인)│    │F1(P,R,F1) │    │(청킹 품질)│             │
│   └───────────┘    └───────────┘    └───────────┘             │
│                                                                  │
│   논리 흐름: CER/WER 통과 → VLM 구조화 → Structure F1 측정      │
│              → BC/CS로 다운스트림 효과 확인                      │
└─────────────────────────────────────────────────────────────────┘
```

## 3.2 Parser Implementations

### 3.2.1 Text-Baseline (PyMuPDF)

**Architecture**: Rule-based PDF text extraction using PyMuPDF (`fitz`)

**Process**:
1. PyMuPDF로 PDF 텍스트 레이어에서 직접 추출
2. `find_tables()`로 테이블 검출 및 추출
3. 텍스트와 테이블 병합

**Code Flow**:
```
OCRParser.parse_pdf(pdf_bytes)
  → fitz.open(stream=pdf_bytes)
  → page.get_text() + page.find_tables()
  → OCRResult(content, tables)
```

**Limitations**:
- 디지털 PDF 전용 (스캔 PDF에서 텍스트 추출 불가)
- 테이블 검출이 라인 규칙에 의존
- 마크다운 구조 요소 생성 불가 (Structure F1 = 0%)

### 3.2.2 Image-Baseline (RapidOCR)

**Architecture**: OCR 기반 이미지 텍스트 추출

**Process**:
1. PDF 페이지를 PNG 이미지로 변환 (300 DPI)
2. RapidOCR로 텍스트 영역 검출 및 인식
3. 텍스트 결과 병합

**Code Flow**:
```
RapidOCRParser.parse_pdf(pdf_bytes)
  → ImageOCRParser.pdf_to_images(pdf_bytes)
  → rapidocr(image) → OCRResult(content)
```

**Limitations**:
- 레이아웃 분석 없음 (다단 문서에서 읽기 순서 오류)
- 마크다운 구조 요소 생성 불가 (Structure F1 = 0%)

### 3.2.3 Text-Advanced (PyMuPDF + VLM)

**Architecture**: Two-Stage Pipeline — PyMuPDF 추출 후 VLM 구조화

**Process**:
1. **Stage 1**: PyMuPDF로 텍스트 추출 (Text-Baseline과 동일)
2. **Stage 2**: VLM(Qwen3-VL-2B-Instruct)에 추출 텍스트 전달 → 마크다운 구조화

**Code Flow**:
```
TwoStageParser.parse_text_pdf(pdf_bytes)
  → Stage 1: OCRParser.parse_pdf(pdf_bytes) → raw_text
  → Stage 2: TextStructurer.structure(raw_text)
    → POST /v1/chat/completions (system + user prompt)
    → structured_markdown
```

### 3.2.4 Image-Advanced (RapidOCR + VLM)

**Architecture**: Two-Stage Pipeline — RapidOCR 추출 후 VLM 구조화

**Process**:
1. **Stage 1**: RapidOCR로 이미지 기반 텍스트 추출 (Image-Baseline과 동일)
2. **Stage 2**: VLM(Qwen3-VL-2B-Instruct)에 추출 텍스트 전달 → 마크다운 구조화

**Code Flow**:
```
TwoStageParser.parse_image_pdf(pdf_bytes)
  → Stage 1: RapidOCRParser.parse_pdf(pdf_bytes) → raw_text
  → Stage 2: TextStructurer.structure(raw_text)
    → POST /v1/chat/completions (system + user prompt)
    → structured_markdown
```

### 3.2.5 입력/출력 예시 (동일 텍스트, 4개 파서 비교)

다음은 "Attention Is All You Need" 논문의 첫 번째 섹션을 각 파서로 처리한 결과 스니펫이다:

**원본 (GT)**:
```markdown
# Attention Is All You Need

## 1. Introduction

Recurrent neural networks, long short-term memory...
```

**Text-Baseline** (PyMuPDF):
```
Attention Is All You Need
1 Introduction
Recurrent neural networks, long short-term memory...
```
→ 텍스트 내용은 유지되나 구조 요소(`#`) 없음

**Image-Baseline** (RapidOCR):
```
Attention Is All You Need
1 Introduction
Recurrent neural networks, long short-term memory...
```
→ OCR 추출 결과. 구조 요소 없음, 간혹 문자 인식 오류 발생

**Text-Advanced** (PyMuPDF + VLM):
```markdown
# Attention Is All You Need

## 1. Introduction

Recurrent neural networks, long short-term memory...
```
→ VLM이 마크다운 구조 요소(`#`, `##`) 생성. 텍스트 일부 변형 가능

**Image-Advanced** (RapidOCR + VLM):
```markdown
# Attention Is All You Need

## 1. Introduction

Recurrent neural networks, long short-term memory...
```
→ OCR 추출 + VLM 구조화. OCR 오류 + VLM 변형 누적 가능

### 3.2.6 정규화(Normalization) 전/후 비교

CER/WER 계산 시 마크다운 문법이 불공정한 차이를 만들지 않도록 정규화를 적용한다:

**정규화 전** (Advanced 파서 출력):
```markdown
## 1. Introduction

Recurrent neural networks, long short-term memory...
```

**정규화 후** (CER/WER 계산용):
```
1. Introduction
Recurrent neural networks, long short-term memory...
```

이 정규화에도 불구하고 Advanced 파서의 CER이 높아지는 이유:
- VLM이 텍스트를 재구성하면서 단어 순서 변경, 추가/삭제 발생
- 특히 다단 문서에서 VLM의 읽기 순서 해석이 원본과 다를 수 있음
- 표 형식 변환 시 셀 내용의 미세한 변형

## 3.3 Evaluation Metrics

### 3.3.1 RQ1: 전제 조건 검증 (CER, WER)

CER/WER은 VLM의 성능 평가 지표가 아니라, 파싱 파이프라인의 **기본 텍스트 추출 품질을 확인하는 전제 조건 지표**이다.

- **Baseline 파서의 CER/WER**: Stage 1 추출 품질이 적정 기준 이상인지 확인
- **Advanced 파서의 CER/WER**: VLM 구조화 과정에서 텍스트 손실/변형이 허용 범위 내인지 확인

#### Character Error Rate (CER)

Measures character-level accuracy using Levenshtein edit distance:

$$CER = \frac{S + D + I}{N}$$

Where:
- $S$ = Number of character substitutions
- $D$ = Number of character deletions
- $I$ = Number of character insertions
- $N$ = Total characters in reference

**Implementation**: Using `jiwer` library with custom preprocessing

#### Word Error Rate (WER)

Measures word-level accuracy with appropriate tokenization:

$$WER = \frac{S_w + D_w + I_w}{N_w}$$

**Tokenization Options**:
| Language | Tokenizer | Rationale |
|----------|-----------|-----------|
| Korean | MeCab | Morphological analysis for agglutinative language |
| English | Whitespace | Space-separated tokens |
| Mixed | MeCab + fallback | Primary Korean, whitespace for others |

### 3.3.2 RQ2: 구조 보존 측정 (Structure F1)

Structure F1은 파서가 Ground Truth의 마크다운 구조 요소를 얼마나 정확하게 재현하는지 측정한다.

#### 구조 요소 정의

GT 마크다운 파일에서 다음 패턴을 구조 요소로 추출한다:

| Element Type | Regex Pattern | Example |
|--------------|---------------|---------|
| Heading | `^#{1,6}\s+` | `# Title`, `## Section` |
| Unordered List | `^[\s]*[-*+]\s+` | `- item` |
| Ordered List | `^[\s]*\d+\.\s+` | `1. first` |
| Table Row | `^\|.+\|$` | `\| col1 \| col2 \|` |
| Code Block | `` ^```  `` | `` ```python `` |

#### Precision, Recall, F1

Structure F1을 이해하려면 먼저 Precision과 Recall을 정의해야 한다:

- **Precision** (정밀도): 파서가 생성한 구조 요소 중 GT에도 존재하는 비율
  $$Precision = \frac{TP}{TP + FP}$$
  → "과잉 생성(False Positive) 여부"를 측정. VLM은 hallucination으로 존재하지 않는 구조를 생성할 수 있으므로 Precision 모니터링이 중요하다.

- **Recall** (재현율): GT의 구조 요소 중 파서가 검출한 비율
  $$Recall = \frac{TP}{TP + FN}$$
  → "누락(False Negative) 여부"를 측정. 구조 요소를 빠뜨리면 청킹 품질이 하락한다.

- **F1 Score**: Precision과 Recall의 조화평균
  $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**왜 F1만이 아니라 Precision/Recall 분리가 필요한가?**:
> F1만으로는 파서가 구조를 과잉 생성(FP)하는지, 누락(FN)하는지 구분할 수 없다. VLM은 hallucination으로 과잉 생성할 수 있으므로 Precision 모니터링이 중요하다. 반대로 Recall이 낮으면 핵심 구조가 누락되어 청킹 품질이 저하된다. 두 지표를 분리 관찰해야 오류 원인을 진단할 수 있다.

### 3.3.3 RQ3: 다운스트림 효과 (BC, CS)

#### Boundary Coherence (BC)

Measures alignment between predicted and ground truth semantic boundaries:

$$BC = \frac{|B_{pred} \cap B_{gt}|}{|B_{gt}|}$$

Where:
- $B_{pred}$ = Set of predicted chunk boundaries
- $B_{gt}$ = Set of ground truth semantic boundaries

**Boundary Definition**: Points where semantic context shifts (section changes, topic transitions)

**Implementation**:
1. Identify structural markers (headers, blank lines, section breaks)
2. Generate boundary positions for both reference and candidate
3. Calculate intersection with tolerance window (±n characters)

#### Chunk Score (CS)

Measures semantic coherence within each chunk:

$$CS = \frac{1}{|C|} \sum_{c \in C} coherence(c)$$

Where:
- $C$ = Set of generated chunks
- $coherence(c)$ = Semantic coherence of chunk $c$ (embedding similarity variance)

**Implementation**:
1. Generate embeddings for sentences within each chunk
2. Calculate intra-chunk similarity variance
3. Lower variance = higher coherence = better score

## 3.4 Normalization Process

To ensure fair comparison, we apply normalization before metric calculation:

### 3.4.1 Markdown Stripping

Remove markdown syntax for lexical comparison:
- Headers: `# Header` → `Header`
- Bold/Italic: `**bold**` → `bold`
- Links: `[text](url)` → `text`
- Tables: Preserve cell content, remove pipes

### 3.4.2 Whitespace Normalization

Standardize whitespace handling:
- Collapse multiple spaces to single space
- Normalize newlines (CRLF → LF)
- Trim leading/trailing whitespace

### 3.4.3 Unicode Normalization

Apply NFKC normalization for consistent character representation:
- Full-width → half-width
- Compatibility characters normalized

## 3.5 Experimental Design

### 3.5.1 A/B Comparison Framework

```
┌─────────────────────────────────────────────────────────────────┐
│  Experiment A (Baseline)                                         │
│  ─────────────────────────────────────────────────────────────  │
│  PDF → PyMuPDF/RapidOCR → Plain Text → Chunking → Retrieval     │
│                                                      ↓          │
│                                              CER/WER + F1=0%    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Experiment B (Advanced = Two-Stage)                             │
│  ─────────────────────────────────────────────────────────────  │
│  PDF → PyMuPDF/RapidOCR → VLM 구조화 → Markdown → Chunking      │
│                                                      ↓          │
│                                         CER/WER + F1 측정       │
└─────────────────────────────────────────────────────────────────┘

논리: RQ1(CER/WER 전제) → RQ2(Structure F1) → RQ3(BC/CS)
```

### 3.5.2 Variable Control

**Fixed Variables** (identical across experiments):
- Chunking algorithm: RecursiveCharacterTextSplitter
- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Embedding model: ko-sroberta-multitask
- Retrieval method: Cosine similarity

**Independent Variable**:
- Parser type: Text-Baseline, Image-Baseline, Text-Advanced, Image-Advanced

**Dependent Variables**:
- CER, WER, Structure F1 (Precision, Recall), BC, CS

## 3.6 Statistical Analysis

### 3.6.1 Significance Testing

- **Paired t-test**: Compare VLM vs Baseline on same documents
- **Wilcoxon signed-rank**: Non-parametric alternative if normality violated

### 3.6.2 Effect Size

- **Cohen's d**: Standardized mean difference
  - Small: d = 0.2
  - Medium: d = 0.5
  - Large: d = 0.8

### 3.6.3 Confidence Intervals

- **Bootstrap 95% CI**: 1000 resamples for robust estimation
- Report: Mean ± 95% CI for all metrics
