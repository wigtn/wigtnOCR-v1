# Abstract

## Research Summary

This technical report presents a comprehensive evaluation of Vision-Language Models (VLMs) for document parsing in Retrieval-Augmented Generation (RAG) pipelines. We investigate whether structured markdown output from VLMs (specifically Qwen3-VL) improves semantic chunking quality and downstream retrieval performance compared to traditional OCR methods (pdfplumber, Docling+RapidOCR).

## Problem Statement

Traditional OCR pipelines often fail to preserve document structure (tables, headers, multi-column layouts), leading to degraded semantic chunking and retrieval accuracy. This study quantifies the impact of structural preservation on RAG system performance.

## Methodology

We employ a multi-phase evaluation framework:
1. **Lexical Accuracy**: Character Error Rate (CER) and Word Error Rate (WER) with Korean morphological analysis
2. **Structural Integrity**: Boundary Score (BS) and Chunk Score (CS) metrics
3. **Retrieval Performance**: Hit Rate@k and Mean Reciprocal Rank (MRR) on generated Q&A pairs

## Key Findings

- **CER/WER Results**: Image-Baseline achieves the lowest CER (40.79%) for pure text extraction. However, VLM-based Advanced parsers prioritize structural preservation over character-level accuracy, resulting in CER increases of 13-24 percentage points.
- **Structural Analysis**: Structure F1 improves dramatically from 0% (Baseline) to 79.25% (Text-Advanced), with Recall of 87.5% and Precision of 72.41%. Advanced parsers successfully detect 21 out of 24 structural elements (TP=21, FN=3).
- **Trade-off Discovery**: A clear trade-off exists between lexical fidelity and structural preservation. Baseline parsers excel at character extraction but lose all structure. Advanced parsers preserve document hierarchy at the cost of some text accuracy.
- **Hallucination Warning**: Korean scanned documents (test_1) showed extreme hallucination with CER of 536% when VLM attempted to "interpret" rather than transcribe content.

## Conclusion

Our findings demonstrate that **structural preservation is achievable** with VLM-based parsing (Structure F1: 0% → 79.25%), but comes with trade-offs in character-level accuracy. We recommend a **hybrid parsing strategy**:

1. **For structure-critical documents** (tables, multi-column): Use Advanced parsers despite higher CER
2. **For text-extraction tasks**: Use Image-Baseline for optimal CER (40.79%)
3. **For Korean scanned documents**: Exercise caution with VLM to avoid hallucination

The 159x latency increase (0.27s → 42.92s) for Advanced parsers is justified when document structure is essential for downstream RAG applications.

---

**Keywords**: Vision-Language Models, Document Parsing, RAG, Semantic Chunking, OCR, Qwen-VL

**Word Count**: ~200 words (target)
