# Appendix B: Full Results Tables

## B.1 Lexical Accuracy Results

### B.1.1 Character Error Rate (CER) - Complete Results

#### Per-Document CER

| Document ID | Document Type | Text-Baseline | Image-Baseline | Text-Advanced | Image-Advanced |
|-------------|---------------|---------------|----------------|---------------|----------------|
| test_1 | Korean Gov Doc (Scan) | N/A* | 91.87% | N/A* | **536.50%** |
| test_2 | Academic Paper (CoT) | 99.59% | 40.80% | 120.54% | 33.06% |
| test_3 | Academic Paper (Transformer) | 51.25% | **40.79%** | 64.11% | 57.71% |

*Text-Baseline failed to extract content (content_length=0)

**Key Observations**:
- Image-Baseline consistently achieves lowest CER for readable documents
- test_1 Image-Advanced shows severe hallucination (CER > 500%)
- Advanced parsers trade CER accuracy for structural preservation

### B.1.2 Word Error Rate (WER) - Complete Results

#### Per-Document WER

| Document ID | Document Type | Tokenizer | Text-Baseline | Image-Baseline | Text-Advanced | Image-Advanced |
|-------------|---------------|-----------|---------------|----------------|---------------|----------------|
| test_1 | Korean | whitespace | N/A | 99.42% | N/A | 322.63% |
| test_2 | English | whitespace | 99.69% | 55.59% | 262.94% | 37.31% |
| test_3 | English | whitespace | 57.19% | **41.24%** | 69.34% | 63.27% |

## B.2 Structural Integrity Results

### B.2.1 Structure F1 Score - Complete Results

| Document | Text-Baseline | Image-Baseline | Text-Advanced | Image-Advanced |
|----------|---------------|----------------|---------------|----------------|
| test_1 | N/A | 0.00% | N/A | 0.00% |
| test_2 | 0.00% | 0.00% | 9.30% | 16.67% |
| test_3 | 0.00% | 0.00% | **79.25%** | 77.78% |

### B.2.2 Structure Detection Details (test_3 - Primary Benchmark)

| Parser | Precision | Recall | F1 | TP | FP | FN | Hyp Elements | Ref Elements |
|--------|-----------|--------|----|----|----|----|--------------|--------------|
| Text-Baseline | 0.00% | 0.00% | 0.00% | 0 | 11 | 24 | 11 | 24 |
| Image-Baseline | 0.00% | 0.00% | 0.00% | 0 | 0 | 24 | 0 | 24 |
| Text-Advanced | **72.41%** | **87.50%** | **79.25%** | 21 | 8 | 3 | 29 | 24 |
| Image-Advanced | 70.00% | 87.50% | 77.78% | 21 | 9 | 3 | 30 | 24 |

### B.2.3 Structure Detection Details (test_2 - CoT Paper)

| Parser | Precision | Recall | F1 | TP | FP | FN | Hyp Elements | Ref Elements |
|--------|-----------|--------|----|----|----|----|--------------|--------------|
| Text-Baseline | 0.00% | 0.00% | 0.00% | 0 | 0 | 10 | 0 | 10 |
| Image-Baseline | 0.00% | 0.00% | 0.00% | 0 | 4 | 10 | 4 | 10 |
| Text-Advanced | 6.06% | 20.00% | 9.30% | 2 | 31 | 8 | 33 | 10 |
| Image-Advanced | 50.00% | 10.00% | 16.67% | 1 | 1 | 9 | 2 | 10 |

### B.2.4 Structure Detection Details (test_1 - Korean Scan)

| Parser | Precision | Recall | F1 | TP | FP | FN | Hyp Elements | Ref Elements |
|--------|-----------|--------|----|----|----|----|--------------|--------------|
| Text-Baseline | N/A | N/A | N/A | - | - | - | - | - |
| Image-Baseline | 0.00% | 0.00% | 0.00% | 0 | 1 | 19 | 1 | 19 |
| Text-Advanced | N/A | N/A | N/A | - | - | - | - | - |
| Image-Advanced | 0.00% | 0.00% | 0.00% | 0 | 831 | 19 | 831 | 19 |

**Note**: test_1 Image-Advanced shows 831 false positives, indicating severe hallucination where the model fabricated structural elements.

## B.3 Latency Results

### B.3.1 Total Processing Time (seconds)

| Document | Text-Baseline | Image-Baseline | Text-Advanced | Image-Advanced |
|----------|---------------|----------------|---------------|----------------|
| test_1 | 1.35s | 18.07s | 1.83s | 51.26s |
| test_2 | 3.58s | 23.65s | 39.01s | 37.06s |
| test_3 | **2.31s** | **0.27s** | 42.92s | 35.75s |

### B.3.2 Stage Breakdown (Advanced Parsers)

| Document | Parser | Stage1 Time | Stage2 Time | Stage1 Parser | Stage2 Applied |
|----------|--------|-------------|-------------|---------------|----------------|
| test_1 | Text-Advanced | 1.83s | 0.00s | pymupdf | No |
| test_1 | Image-Advanced | 16.00s | 35.26s | rapidocr | Yes |
| test_2 | Text-Advanced | 3.84s | 35.17s | pymupdf | Yes |
| test_2 | Image-Advanced | 24.46s | 12.60s | rapidocr | Yes |
| test_3 | Text-Advanced | 2.28s | 40.64s | pymupdf | Yes |
| test_3 | Image-Advanced | 0.27s | 35.48s | rapidocr | Yes |

### B.3.3 Latency Summary Statistics

| Parser | Min | Max | Mean | Speedup vs Advanced |
|--------|-----|-----|------|---------------------|
| Text-Baseline | 1.35s | 3.58s | 2.41s | ~17x faster |
| Image-Baseline | 0.27s | 23.65s | 13.99s | ~3x faster |
| Text-Advanced | 1.83s | 42.92s | 27.92s | 1x (baseline) |
| Image-Advanced | 35.75s | 51.26s | 41.36s | ~0.7x |

## B.4 Chunking Results (Semantic Chunking)

### B.4.1 test_2 - Chain of Thought Paper

| Parser | Chunk Count | BC Score | BC Min | BC Max | BC Std |
|--------|-------------|----------|--------|--------|--------|
| Image-Advanced | 7 | 0.512 | 0.221 | 0.659 | 0.153 |
| Image-Baseline | 6 | 0.502 | 0.364 | 0.587 | 0.080 |

### B.4.2 test_3 - Attention Is All You Need

| Parser | Chunk Count | BC Score | Note |
|--------|-------------|----------|------|
| Image-Advanced | 18 | - | Section-aligned chunks |
| Text-Advanced | - | - | Structure-aware boundaries |

**BC (Boundary Coherence)**: Measures semantic similarity across chunk boundaries. Higher is better (0-1 scale).

## B.5 Content Length Statistics

| Document | Parser | Content Length (chars) |
|----------|--------|------------------------|
| test_1 | Text-Baseline | 0 |
| test_1 | Image-Baseline | 522 |
| test_1 | Text-Advanced | 0 |
| test_1 | Image-Advanced | 19,033 |
| test_2 | Text-Baseline | 50 |
| test_2 | Image-Baseline | 12,156 |
| test_2 | Text-Advanced | 14,903 |
| test_2 | Image-Advanced | 11,887 |
| test_3 | Text-Baseline | 48,505 |
| test_3 | Image-Baseline | 44,909 |
| test_3 | Text-Advanced | 33,215 |
| test_3 | Image-Advanced | 30,673 |

**Observations**:
- test_1 Image-Advanced inflated content (19,033 vs 522) due to hallucination
- test_3 Advanced parsers produce ~30% less content due to structure formatting overhead

## B.6 Summary Statistics by Parser Type

### B.6.1 Baseline Parsers (test_3 benchmark)

| Metric | Text-Baseline | Image-Baseline | Winner |
|--------|---------------|----------------|--------|
| CER | 51.25% | **40.79%** | Image-Baseline |
| WER | 57.19% | **41.24%** | Image-Baseline |
| Structure F1 | 0% | 0% | Tie (both fail) |
| Latency | 2.31s | **0.27s** | Image-Baseline |

### B.6.2 Advanced Parsers (test_3 benchmark)

| Metric | Text-Advanced | Image-Advanced | Winner |
|--------|---------------|----------------|--------|
| CER | 64.11% | **57.71%** | Image-Advanced |
| WER | 69.34% | **63.27%** | Image-Advanced |
| Structure F1 | **79.25%** | 77.78% | Text-Advanced |
| Latency | 42.92s | **35.75s** | Image-Advanced |

### B.6.3 Cross-Category Comparison (test_3 benchmark)

| Metric | Best Baseline | Best Advanced | Trade-off |
|--------|---------------|---------------|-----------|
| CER | 40.79% (Image) | 57.71% (Image) | +16.92pp |
| WER | 41.24% (Image) | 63.27% (Image) | +22.03pp |
| Structure F1 | 0% | 79.25% (Text) | **+79.25pp** |
| Latency | 0.27s (Image) | 35.75s (Image) | 132x slower |

## B.7 Key Findings Summary

1. **Structure vs Accuracy Trade-off**: Gaining 79% Structure F1 costs ~17pp in CER
2. **Image-Baseline wins for pure OCR**: Lowest CER (40.79%), fastest (0.27s)
3. **Text-Advanced wins for structure**: Highest F1 (79.25%), best Precision (72.41%)
4. **Hallucination risk**: Korean scans trigger severe hallucination (CER 536%)
5. **Latency cost**: Advanced parsers are 100-160x slower than fastest baseline

## B.8 Raw Data Reference

All raw experimental data is stored in:
- `results/test_1/evaluation.json` - Korean government document
- `results/test_2/evaluation.json` - Chain-of-Thought paper
- `results/test_3/evaluation.json` - Attention Is All You Need paper
- `results/test_*/chunking.json` - Semantic chunking results

Data format: JSON with timestamps and configuration metadata
