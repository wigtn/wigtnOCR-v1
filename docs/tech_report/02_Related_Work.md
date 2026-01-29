# 2. Related Work

## 2.1 Document Understanding and Layout Analysis

### 2.1.1 Traditional OCR Systems

Traditional OCR pipelines have evolved significantly over the past decades:

- **Tesseract** (Smith, 2007): Open-source OCR engine with LSTM-based recognition
- **Adobe Acrobat**: Commercial solution with layout-aware text extraction
- **ABBYY FineReader**: Enterprise-grade OCR with document structure recognition

**Limitations**: These systems primarily focus on character recognition accuracy and often struggle with:
- Complex table structures
- Multi-column reading order
- Semantic relationship preservation

### 2.1.2 PDF Text Extraction Libraries

Several Python libraries specialize in PDF text extraction:

- **pdfplumber** (Singer-Vine, 2022): Precise positional text extraction with table detection
- **PyMuPDF (fitz)**: Fast PDF rendering and text extraction
- **pdfminer.six**: Layout-aware text extraction

**Trade-offs**:
| Library | Speed | Table Support | Scanned PDF |
|---------|-------|---------------|-------------|
| pdfplumber | Medium | Good | None |
| PyMuPDF | Fast | Basic | None |
| pdfminer | Slow | Basic | None |

### 2.1.3 Layout-Aware Models

Recent advances in document understanding leverage transformer architectures:

- **LayoutLM** (Xu et al., 2020): Pre-trained model combining text and layout information
- **LayoutLMv2** (Xu et al., 2021): Adds visual features from document images
- **LayoutLMv3** (Huang et al., 2022): Unified text-image-layout pre-training

**Application**: These models excel at document classification and information extraction but require fine-tuning for specific tasks.

## 2.2 Vision-Language Models for Document Parsing

### 2.2.1 General-Purpose VLMs

Large Vision-Language Models have demonstrated strong document understanding capabilities:

- **GPT-4V** (OpenAI, 2023): Multimodal model with document parsing capabilities
- **Claude 3** (Anthropic, 2024): Strong document analysis with structured output
- **Gemini Pro Vision** (Google, 2024): Multimodal understanding across formats

### 2.2.2 Document-Specialized VLMs

Models specifically designed for document understanding:

- **Qwen-VL** (Bai et al., 2023): Open-source VLM with strong Chinese/English support
- **Qwen2-VL** (Wang et al., 2024): Enhanced visual understanding and instruction following
- **Qwen3-VL** (2025): Improved reasoning and structured output generation
- **Docling** (IBM, 2024): Document-specialized parsing with OCR integration
- **Nougat** (Blecher et al., 2023): Academic paper to markdown conversion

### 2.2.3 Prompt Engineering for Document Parsing

The effectiveness of VLMs depends heavily on prompt design:

- **Extraction-focused prompts**: "Extract all information from this document"
  - Risk: Hallucination, content addition
- **Transcription-focused prompts**: "Transcribe exactly what you see"
  - Benefit: Faithful reproduction, reduced hallucination
- **Structured prompts**: Specify output format (markdown, JSON, XML)

#### 프롬프트 진화 히스토리 (본 연구)

본 연구에서는 프롬프트를 반복 실험을 통해 개선했다:

1. **v1 (초기)**: 일반적인 "document structure expert" 프롬프트
   - 접근: "You are an expert document extraction assistant"
   - 결과: Structure F1 = **0%** (헤딩 마커 `#` 미생성)
   - 문제: 2B 파라미터 소형 모델이 암시적 지시("organized markdown")만으로는 마크다운 구조 요소를 생성하지 못함

2. **v2 (commit `90d516d`)**: CRITICAL RULES + 명시적 헤딩 레벨 매핑
   - 변경사항:
     - System/User 프롬프트 분리 (역할 명확화)
     - "MUST", "NEVER" 강제 지시어 추가
     - 번호 체계 → 마크다운 레벨 매핑 규칙 명시 (1→`##`, 2.1→`###`, 3.1.1→`####`)
   - 결과: Structure F1 = **0% → ~79%** (test_3 기준)
   - 핵심 개선: 모델이 "ALWAYS use # symbols" 규칙을 따르기 시작

3. **교훈**: 2B 규모의 소형 모델에서는 암시적 지시("clean markdown")보다 **명시적 규칙**("MUST use #", "1→##, 2.1→###")이 효과적이다. v1→v2 전환은 test_3의 Structure F1 0%를 관찰한 후, 헤딩 미생성 원인을 분석하여 결정되었다.

## 2.3 Semantic Chunking and RAG

### 2.3.1 Chunking Strategies

Text chunking approaches for RAG systems:

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Fixed-size** | Split at character/token count | Simple, predictable | Breaks semantic units |
| **Sentence-based** | Split at sentence boundaries | Preserves sentences | Variable sizes |
| **Semantic** | Split at topic shifts | Meaningful chunks | Complex, slow |
| **Hierarchical** | Nested chunks by structure | Preserves hierarchy | Requires structure |

### 2.3.2 Structure-Aware Chunking

Recent work explores structure-aware chunking:

- **Markdown-based**: Use headers and sections as natural boundaries
- **Table preservation**: Keep tables as atomic units
- **List handling**: Preserve list coherence

**Key Insight**: VLM-generated markdown naturally provides these structural cues.

### 2.3.3 Retrieval Evaluation

Standard metrics for retrieval evaluation:

- **Hit Rate@k**: Whether relevant document appears in top-k results
- **MRR (Mean Reciprocal Rank)**: Average inverse rank of first relevant result
- **NDCG**: Normalized discounted cumulative gain
- **Recall@k**: Proportion of relevant documents retrieved

## 2.4 Evaluation Metrics for Document Parsing

### 2.4.1 Lexical Metrics

- **CER (Character Error Rate)**: Edit distance at character level
  ```
  CER = (S + D + I) / N
  ```
  Where S=substitutions, D=deletions, I=insertions, N=reference length

- **WER (Word Error Rate)**: Edit distance at word/token level
  ```
  WER = (S + D + I) / N
  ```
  Requires appropriate tokenization (morphological for Korean)

### 2.4.2 Structural Metrics

- **BLEU**: N-gram overlap (originally for machine translation)
- **ROUGE**: Recall-oriented overlap (originally for summarization)
- **Boundary Score**: Alignment of semantic boundaries (this work)
- **Chunk Score**: Semantic coherence within chunks (this work)

### 2.4.3 Application-Level Metrics

- **Retrieval accuracy**: Direct measurement of downstream task performance
- **Answer quality**: End-to-end RAG evaluation (RAGAs framework)

## 2.5 Gap Analysis

Despite advances in document parsing:

1. **Limited evaluation frameworks**: Most work focuses on single metrics (accuracy or retrieval)
2. **Missing structural analysis**: Few studies quantify structural preservation impact
3. **Lack of Korean support**: Most benchmarks focus on English documents
4. **Cost-quality trade-offs**: No systematic analysis of when VLM is worth the cost

This study addresses these gaps with a comprehensive multi-metric evaluation framework.

## References

<!-- TODO: Add full citations -->
- Xu et al. (2020). LayoutLM: Pre-training of Text and Layout for Document Image Understanding
- Bai et al. (2023). Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
- Blecher et al. (2023). Nougat: Neural Optical Understanding for Academic Documents
