# 4. Experimental Setup

## 4.1 Dataset Description

### 4.1.1 Test Documents Overview

| ID | Document Type | Language | Pages | Complexity | Source |
|----|---------------|----------|-------|------------|--------|
| test_1 | Government Announcement | Korean | 1 | Tables, Headers | Public AI Support Program |
| test_2 | Receipt Image | Korean | 1 | Structured Text | Restaurant Receipt |
| test_3 | Academic Paper | English | Multi | Multi-column, References | arXiv (CoT Prompting) |

### 4.1.2 Test Case Details

#### Test 1: Korean Government Document (test_data_1.pdf)

**Characteristics**:
- Digital PDF with embedded text
- Complex table structures
- Multiple sections with headers
- Mix of Korean text and numbers
- Official document formatting

**Ground Truth Focus**:
- Table structure preservation
- Header hierarchy
- List formatting
- Number accuracy

#### Test 2: Receipt Image (test_data_2.jpg)

**Characteristics**:
- Scanned/photographed document
- Structured layout (merchant info, items, totals)
- Mixed fonts and sizes
- Korean and numbers

**Ground Truth Focus**:
- Text extraction accuracy
- Line item preservation
- Amount accuracy

#### Test 3: Academic Paper (Chain-of-Thought-Prompting.pdf)

**Characteristics**:
- Multi-page academic PDF
- Two-column layout
- Figures and tables
- References section
- Mathematical notation

**Ground Truth Focus**:
- Reading order (column handling)
- Section boundaries
- Figure/table separation
- Reference accuracy

### 4.1.3 Ground Truth Creation

Ground truth files were manually created following these guidelines:

1. **Markdown Format**: Use standard markdown syntax
2. **Structure Preservation**: Maintain headers, lists, tables
3. **Verbatim Text**: No paraphrasing or summarization
4. **Completeness**: Include all visible text
5. **Format Consistency**: Standardized markdown styling

**구조 요소 작성 기준**:
- 문서 제목 → `# Title`
- 번호가 있는 최상위 섹션 (1, 2, 3...) → `## Section`
- 하위 섹션 (2.1, 3.2...) → `### Subsection`
- 하위-하위 섹션 (3.1.1, 4.2.1...) → `#### Sub-subsection`
- 리스트 항목 → `- item` 또는 `1. item`
- 표 → `| col1 | col2 |` with `|---|---|` separator

**Quality Assurance**:
- Manual review by document owner
- Cross-validation against original PDF
- Edge case annotation (unclear text marked)

## 4.2 Q&A Dataset Generation

### 4.2.1 Generation Strategy

Q&A pairs are generated using LLM-assisted creation:

```python
prompt = """
Based on the following document content, generate {n} question-answer pairs.

Requirements:
1. Questions should be answerable from the document
2. Include diverse question types:
   - Factual: "What is X?"
   - Table lookup: "How much is Y?"
   - Multi-hop: "Compare X and Y"
   - Inferential: "Why does X happen?"
3. Provide exact text spans as answers
4. Include difficulty ratings (easy/medium/hard)

Document:
{document_content}
"""
```

### 4.2.2 Question Types

| Type | Description | Example | Target |
|------|-------------|---------|--------|
| Factual | Direct fact retrieval | "What is the support period?" | Single chunk |
| Table Lookup | Information from tables | "What is the maximum funding?" | Table cell |
| Multi-hop | Combine multiple facts | "Compare A and B programs" | Multiple chunks |
| Inferential | Reasoning required | "Why is this program important?" | Implicit info |

### 4.2.3 Dataset Statistics

<!-- TODO: Fill after Q&A generation -->
| Document | Total Q&A | Factual | Table | Multi-hop | Inferential |
|----------|-----------|---------|-------|-----------|-------------|
| test_1 | 10-20 | TBD | TBD | TBD | TBD |
| test_2 | 10-20 | TBD | TBD | TBD | TBD |
| test_3 | 10-20 | TBD | TBD | TBD | TBD |

## 4.3 Parser Configuration

### 4.3.1 Text-Baseline / Image-Baseline Settings

```yaml
text_baseline:
  library: "PyMuPDF (fitz)"
  min_text_length: 50  # 디지털 PDF 판별 기준
  table_detection: "find_tables()"

image_baseline:
  library: "RapidOCR (rapidocr-onnxruntime)"
  image_resolution: 300  # DPI
```

### 4.3.2 Advanced Parser (VLM) Settings

```yaml
vlm_structurer:
  model: "Qwen3-VL-2B-Instruct"
  api_url: "http://localhost:8005/v1/chat/completions"
  temperature: 0.1
  max_tokens: 8192
  prompt_version: "v2"  # CRITICAL RULES + 명시적 헤딩 매핑
```

**System Prompt**:
```
You are a Markdown formatting expert. Your task is to convert plain text
into well-structured Markdown format.

CRITICAL RULES - You MUST follow these:
1. ALWAYS use # symbols for headings. This is mandatory.
2. Document title → # Title
3. Section numbers like "1 Introduction" → ## 1. Introduction
4. Subsections like "3.1 Method" → ### 3.1 Method
5. Sub-subsections like "3.1.1 Details" → #### 3.1.1 Details
6. Tables with aligned columns → Markdown table with | separators
7. Bullet points → - item
8. Numbered lists → 1. item

NEVER output plain text headings without # symbols.
```

**User Prompt**:
```
Format this text as Markdown. Add # symbols to section headings.

HEADING LEVELS (based on numbering depth):
- Paper/document title (no number) → # Actual Title
- Top sections (1, 2, 3...) → ## 1. Section Name
- Subsections (2.1, 3.2...) → ### 2.1 Subsection
- Sub-subsections (3.1.1, 4.2.1...) → #### 3.1.1 Name

OTHER RULES:
- Tables: use | col1 | col2 | with |---|---| separator
- Lists: use - or 1.
- PRESERVE all original text content exactly
- Output Markdown only

TEXT:
{text}

OUTPUT:
```

### 4.3.3 이전 설정과의 차이 (수정 이력)

보고서 초기 버전에서는 다음과 같은 설정이 기술되어 있었으나, 실제 코드와 불일치했다:

| 항목 | 보고서(초기) | 실제(코드) |
|------|-------------|-----------|
| Text-Baseline | pdfplumber | PyMuPDF (`fitz`) |
| Image-Baseline | Docling + RapidOCR | RapidOCR (독립, Docling 미사용) |
| VLM | Qwen3-VL-2B-Instruct | Qwen3-VL-2B-Instruct ✓ |
| API URL | localhost:8000 | localhost:8005 |
| Temperature | 0.0 | 0.1 |
| Max tokens | 4096 | 8192 |

본 보고서는 **실제 코드 기준**으로 수정되었다.

## 4.4 Chunking Configuration

### 4.4.1 Semantic Chunking Parameters

**Controlled variables** (identical for all experiments):

```yaml
chunking:
  strategy: "recursive_character"
  chunk_size: 500
  chunk_overlap: 50
  separators:
    - "\n\n"  # Paragraph break
    - "\n"    # Line break
    - ". "    # Sentence end
    - " "     # Word break
  length_function: "character_count"
```

### 4.4.2 Embedding Configuration

```yaml
embedding:
  model: "jhgan/ko-sroberta-multitask"
  dimension: 768
  normalize: true
  batch_size: 32
```

### 4.4.3 Retrieval Configuration

```yaml
retrieval:
  method: "cosine_similarity"
  top_k: [1, 3, 5, 10]
  threshold: 0.0  # No threshold filtering
```

## 4.5 Evaluation Environment

### 4.5.1 Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition × 2 (각 96GB VRAM) |
| RAM | 128GB DDR5 |
| Storage | SSD |

### 4.5.2 Software

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.13 | Runtime |
| PyMuPDF (fitz) | 1.24.0+ | PDF text extraction (Text-Baseline) |
| RapidOCR | 1.4.0+ | OCR extraction (Image-Baseline) |
| jiwer | 3.0.0+ | CER/WER calculation |
| konlpy | 0.6.0+ | Korean tokenization |
| langchain-experimental | 0.3.0+ | Semantic chunking |
| streamlit | 1.45.0+ | Dashboard |
| plotly | 5.18.0+ | Visualization |

### 4.5.3 Reproducibility

- Random seed: 42 (where applicable)
- All experiments logged with timestamps
- Configuration files versioned
- Results stored in structured JSON format

## 4.6 Experimental Protocol

### 4.6.1 Baseline Measurement

1. Parse documents with PyMuPDF (digital) / RapidOCR (scanned)
2. Apply semantic chunking with fixed parameters
3. Generate embeddings for all chunks
4. Run retrieval for all Q&A pairs
5. Calculate metrics: CER, WER, Structure F1

### 4.6.2 Advanced (Two-Stage) Measurement

1. Parse documents with PyMuPDF/RapidOCR (Stage 1)
2. Apply VLM structuring with Qwen3-VL (Stage 2)
3. Apply identical chunking
4. Generate embeddings (same model)
5. Run retrieval (same queries)
6. Calculate metrics

### 4.6.3 Structural Analysis

1. Compare chunking outputs between parsers
2. Calculate Boundary Coherence (BC)
3. Calculate Chunk Score (CS)
4. Correlate with retrieval performance

## 4.7 Ablation Study Design

### 4.7.1 Prompt Variation

| Version | Approach | Hypothesis |
|---------|----------|------------|
| v1 | "extraction expert" | Higher hallucination |
| v2 | "transcription engine" + CRITICAL RULES | Lower hallucination, better structure |
| v3 | Minimal instruction | Baseline behavior |
| v4 | XML structured output | Explicit structure |

### 4.7.2 Resolution Study

| DPI | Image Size | Hypothesis |
|-----|------------|------------|
| 72 | Small | Fast but lower quality |
| 150 | Medium | Balanced |
| 300 | Large | Best quality, slower |

### 4.7.3 Chunking Strategy Study

| Strategy | Description | Hypothesis |
|----------|-------------|------------|
| Fixed (500) | Fixed character count | Simple baseline |
| Semantic | Topic-based splitting | Better coherence |
| Hierarchical | Structure-aware | Preserves sections |
