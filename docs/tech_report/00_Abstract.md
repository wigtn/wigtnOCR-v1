# Abstract

## Research Summary

This technical report presents a comprehensive evaluation of Vision-Language Models (VLMs) for document parsing in Retrieval-Augmented Generation (RAG) pipelines. We investigate whether VLM-based Two-Stage Parsing (OCR extraction → VLM structuring) preserves document structure better than traditional OCR methods, and whether that structural preservation improves downstream semantic chunking quality.

## Problem Statement

Traditional OCR pipelines (PyMuPDF, RapidOCR) extract text but fail to preserve document structure (tables, headers, multi-column layouts), resulting in Structure F1 = 0%. This study quantifies the structural preservation gap and evaluates VLM-based structuring as a solution.

## Research Questions

| RQ | Question | Metrics | Role |
|----|----------|---------|------|
| RQ1 | OCR 추출 품질이 VLM 입력으로 충분한가? | CER, WER | 전제 조건 검증 |
| RQ2 | VLM Two-Stage Parsing이 문서 구조를 더 잘 보존하는가? | Structure F1 (P, R) | 핵심 가설 검증 |
| RQ3 | 구조 보존이 시맨틱 청킹 품질을 향상시키는가? | BC, CS | 다운스트림 효과 |

## Methodology

Four parser variants are compared on 3 test documents:
- **Text-Baseline**: PyMuPDF text extraction
- **Image-Baseline**: RapidOCR image OCR
- **Text-Advanced**: PyMuPDF + Qwen3-VL-2B-Instruct structuring
- **Image-Advanced**: RapidOCR + Qwen3-VL-2B-Instruct structuring

## Key Findings

- **RQ1 (Prerequisite)**: Baseline CER 40-51% confirms sufficient text extraction quality for VLM input. However, Korean scanned documents showed hallucination (CER 536%), indicating VLM applicability limits.
- **RQ2 (Core Result)**: Structure F1 improves dramatically from **0% → 79.25%** (Text-Advanced). Recall 87.5% (21/24 elements detected), Precision 72.41%. Prompt evolution from v1 (generic) to v2 (CRITICAL RULES + explicit heading mapping) was the key enabler.
- **RQ3 (Downstream)**: Boundary Coherence (BC) score of 0.512 with 18 natural chunk divisions suggests structural preservation benefits chunking quality.
- **Trade-off**: VLM structuring gains +79pp Structure F1 at the cost of +17pp CER increase and 159× latency.

## Conclusion

VLM-based Two-Stage Parsing achieves **structural preservation** (F1: 0% → 79%) that traditional OCR cannot provide. CER/WER serves as a prerequisite check, not the primary evaluation metric for VLM structuring. We recommend a **hybrid parsing strategy** based on document complexity, guided by a three-stage evaluation framework: CER/WER → Structure F1 → BC/CS.

---

**Keywords**: Vision-Language Models, Document Parsing, RAG, Semantic Chunking, Structure F1, Two-Stage Parsing, Qwen-VL

**Experimental Environment**: Qwen3-VL-2B-Instruct, NVIDIA RTX PRO 6000 Blackwell × 2, PyMuPDF + RapidOCR
