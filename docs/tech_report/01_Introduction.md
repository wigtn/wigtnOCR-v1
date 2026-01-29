# 1. Introduction

## 1.1 Problem Definition

Retrieval-Augmented Generation (RAG) systems rely on accurate document parsing to create meaningful text chunks for semantic search. However, traditional Optical Character Recognition (OCR) methods often fail to preserve critical structural elements:

- **Table structures** collapse into unreadable text streams
- **Multi-column layouts** produce incorrect reading order
- **Header hierarchies** lose semantic relationships
- **Lists and enumerations** merge into continuous paragraphs

These structural failures propagate through the RAG pipeline, degrading both chunking quality and retrieval accuracy.

> **구조화된 데이터 정의**: 본 보고서에서 "구조화된 데이터"란 마크다운 구조 요소(Heading `#`, List `-`/`1.`, Table `|...|`, Code Block `` ``` ``)가 포함된 텍스트를 의미한다. Ground Truth(GT) 마크다운 파일에 포함된 구조 요소 수를 기준으로 측정하며, Structure F1 Score로 정량화한다.

## 1.2 Research Questions

This study addresses three primary research questions:

### RQ1: 텍스트 추출 품질 전제 검증 (Prerequisite Validation)
> OCR/텍스트 추출의 기본 정확도(CER, WER)가 VLM 구조화 단계로 넘어가기에 충분한 수준인가?

**Metrics**: CER (Character Error Rate), WER (Word Error Rate)
**역할**: VLM 구조화의 **전제 조건** 검증. Baseline 파서의 CER/WER로 Stage 1 추출 품질을 확인하고, Advanced 파서의 CER/WER로 VLM 구조화 과정에서 텍스트 손실/변형이 허용 범위 내인지 확인한다.

### RQ2: 구조 보존 효과 (Structural Preservation)
> VLM 기반 Two-Stage Parsing을 적용하면 문서의 구조를 얼마나 더 잘 보존하는가?

**Metrics**: Structure F1 (Precision, Recall 포함)
**역할**: **핵심 가설 검증**. Baseline 파서 대비 Advanced 파서가 마크다운 구조 요소(Heading, List, Table)를 얼마나 정확하게 생성하는지 측정한다.

### RQ3: 다운스트림 효과 (Downstream Impact)
> 구조 보존이 시맨틱 청킹 품질을 향상시키는가?

**Metrics**: BC (Boundary Coherence), CS (Chunk Score)
**역할**: **다운스트림 효과** 검증. 구조화된 마크다운 출력이 실제 청킹 품질에 미치는 영향을 측정한다.

**논리 흐름**: CER/WER 전제 확인 → VLM 구조화 적용 → Structure F1로 구조화 효과 측정 → BC/CS로 청킹 품질 확인

| RQ | 질문 | 지표 | 역할 |
|----|------|------|------|
| RQ1 | OCR 추출 품질이 VLM 입력으로 충분한가? | CER, WER | 전제 조건 검증 |
| RQ2 | VLM Two-Stage Parsing이 문서 구조를 더 잘 보존하는가? | Structure F1 (Precision, Recall) | 핵심 가설 검증 |
| RQ3 | 구조 보존이 시맨틱 청킹 품질을 향상시키는가? | BC, CS | 다운스트림 효과 |

## 1.3 Core Hypothesis

> "Structured parsing via Vision-Language Models produces better semantic chunks than traditional OCR, resulting in measurable improvements in retrieval accuracy."

This hypothesis posits that even when using identical chunking algorithms, the input quality (structured markdown vs. plain text) significantly affects output quality.

## 1.4 Contributions

This work makes the following contributions:

1. **Evaluation Framework**: A comprehensive multi-metric framework for comparing document parsers, including:
   - Lexical metrics (CER, WER) with Korean morphological analysis
   - Structural metrics (Structure F1, Boundary Coherence, Chunk Score)
   - Retrieval metrics (Hit Rate, MRR)

2. **Empirical Analysis**: Quantitative comparison of four parsing approaches:
   - Text-Baseline: PyMuPDF 기반 디지털 PDF 텍스트 추출
   - Image-Baseline: RapidOCR 기반 이미지 OCR
   - Text-Advanced: PyMuPDF + VLM(Qwen3-VL) 구조화
   - Image-Advanced: RapidOCR + VLM(Qwen3-VL) 구조화

3. **Hybrid Strategy**: Data-driven recommendations for when to use VLM vs. traditional OCR based on document characteristics

4. **Error Taxonomy**: Categorized analysis of parsing failures with case studies

5. **Reproducible Benchmark**: Open-source evaluation toolkit with ground truth datasets

## 1.5 Scope and Limitations

### In Scope
- Korean and English documents
- Digital PDFs and scanned documents
- Tables, multi-column layouts, headers, lists
- RAG retrieval evaluation (not full end-to-end answer generation)

### Out of Scope
- Handwritten text recognition
- Complex diagrams and charts
- Real-time streaming applications
- Production deployment optimization

## 1.6 Document Structure

The remainder of this report is organized as follows:

- **Section 2**: Related work in document parsing and VLM applications
- **Section 3**: Methodology and evaluation framework
- **Section 4**: Experimental setup and dataset description
- **Section 5**: Results and analysis
- **Section 6**: Discussion and implications
- **Section 7**: Conclusion and future work
- **Section 8**: References
