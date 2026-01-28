# 6. Discussion

## 6.1 Research Question Analysis

### 6.1.1 RQ1: Lexical Fidelity

**Question**: Does VLM-based parsing achieve better character-level and word-level accuracy compared to traditional OCR methods?

**Finding**: VLM-based Advanced parsers do NOT achieve better lexical accuracy. Instead, they trade character fidelity for structural preservation, resulting in CER increases of 13-24 percentage points.

**Analysis**:

| Parser | CER | WER | Key Observation |
|--------|-----|-----|-----------------|
| Text-Baseline | 51.25% | 57.19% | Moderate accuracy, no structure |
| Image-Baseline | **40.79%** | **41.24%** | Best lexical accuracy |
| Text-Advanced | 64.11% | 69.34% | +13pp CER, gains structure |
| Image-Advanced | 57.71% | 63.27% | +17pp CER, gains structure |

The CER and WER results reveal several patterns:

1. **Digital PDFs (test_3 - Attention Is All You Need)**:
   - Image-Baseline achieves the lowest CER (40.79%) through pure OCR extraction
   - Advanced parsers sacrifice ~17pp CER to achieve 79% Structure F1
   - This trade-off is intentional: VLM prioritizes semantic structure over verbatim transcription

2. **Scanned Documents (test_1 - Korean Government Doc)**:
   - Text-Baseline failed to extract content (content_length=0)
   - Image-Baseline achieved CER 91.87% (poor quality scan)
   - Image-Advanced showed hallucination: CER 536%, generating fabricated content
   - **Critical insight**: VLM hallucination is a severe risk for low-quality Korean scans

3. **Korean Language Impact**:
   - Whitespace tokenization used for consistency across tests
   - Korean documents showed higher error rates across all parsers
   - MeCab tokenization would provide more nuanced WER analysis

### 6.1.2 RQ2: Structural Preservation

**Question**: Does VLM-based parsing preserve document structure better, leading to improved semantic chunking?

**Finding**: **Yes, dramatically.** Structure F1 improves from 0% (Baseline) to 79.25% (Text-Advanced). Advanced parsers detect 21 out of 24 structural elements with 87.5% Recall.

**Analysis**:

| Parser | Structure F1 | Precision | Recall | TP | FP | FN |
|--------|-------------|-----------|--------|----|----|-----|
| Text-Baseline | 0% | 0% | 0% | 0 | 11 | 24 |
| Image-Baseline | 0% | 0% | 0% | 0 | 0 | 24 |
| Text-Advanced | **79.25%** | 72.41% | 87.5% | 21 | 8 | 3 |
| Image-Advanced | 77.78% | 70.0% | 87.5% | 21 | 9 | 3 |

1. **Structure F1 Breakthrough (0% → 79%)**:
   - Baseline parsers completely fail to detect structural elements (headers, tables)
   - Advanced parsers with VLM stage2 achieve near-80% F1 score
   - High Recall (87.5%) means most structures are detected
   - Moderate Precision (72%) indicates some false positives (over-detection)

2. **Error Analysis (FP/FN)**:
   - **False Negatives (3)**: Missed elements are typically subtle formatting or nested structures
   - **False Positives (8-9)**: VLM occasionally over-interprets formatting as structure
   - Net result: Better to over-detect than miss critical structure

3. **Chunking Quality Improvement**:
   - test_2 shows BC (Boundary Coherence) of 0.512 with 18 chunks
   - Markdown headers and sections create natural chunk boundaries
   - Tables remain atomic units, preventing mid-table splits

### 6.1.3 RQ3: Retrieval Impact

**Question**: Does structural preservation in parsing improve downstream retrieval performance in RAG systems?

**Finding**: Indirect evidence suggests structural preservation improves chunking quality, which is a prerequisite for better retrieval. Direct Hit Rate measurement was not conducted in this experiment cycle.

**Analysis**:

1. **Chunking Quality as Proxy**:
   - test_2 Image-Advanced: 7 chunks, BC score 0.512 (moderate coherence)
   - test_3 Image-Advanced: 18 chunks with natural section boundaries
   - Chunks align with document sections (Introduction, Methods, Results, etc.)

2. **Causal Chain (Theoretical)**:
   ```
   Better Parsing → Better Boundaries → Better Chunks → Better Retrieval
   (VLM +79% F1)   (Natural breaks)   (18 sections)   (Expected ↑HR@k)
   ```

3. **Structural Benefits for Retrieval**:
   - Headers become chunk metadata (filterable)
   - Tables remain atomic (no split tables in retrieval)
   - Section hierarchy enables hierarchical retrieval
   - BC score 0.512 indicates moderate semantic coherence between adjacent chunks

4. **Limitations**:
   - Direct Hit Rate@k evaluation not performed in this cycle
   - Future work should measure end-to-end retrieval impact
   - Q&A pairs needed for proper evaluation

## 6.2 Error Pattern Analysis

### 6.2.1 VLM Error Categories (Advanced Parsers)

| Category | Frequency | Severity | Root Cause | Example |
|----------|-----------|----------|------------|---------|
| HALLUCINATION | High (test_1) | **Critical** | Over-interpretation of unclear content | CER 536% on Korean scan |
| FALSE_POSITIVE_STRUCTURE | 27-30% | Medium | Over-detection of structure | FP=8-9 in test_3 |
| MISSED_STRUCTURE | 12.5% | Medium | Subtle formatting not detected | FN=3 in test_3 |
| SUBSTITUTION | Moderate | Minor | Similar characters, OCR artifacts | Contributes to +17pp CER |
| LATENCY | 100% | Varies | VLM inference time | 42.92s vs 0.27s |

### 6.2.2 Traditional OCR Error Categories (Baseline Parsers)

| Category | Frequency | Severity | Root Cause | Example |
|----------|-----------|----------|------------|---------|
| STRUCTURE_LOSS | **100%** | Critical | No layout understanding | Structure F1 = 0% |
| TABLE_COLLAPSE | 100% | Critical | Tables become text streams | All tables flattened |
| COLUMN_MIX | High | Critical | Multi-column ordering | test_3 academic paper |
| EMPTY_EXTRACTION | Varies | Critical | Scanned PDF failure | test_1 Text-Baseline |
| CHARACTER_ERROR | Low-Moderate | Minor | OCR accuracy | CER 40.79% best case |

### 6.2.3 Comparative Error Analysis

**When VLM Wins**:
1. Complex table structures (merged cells, nested headers)
2. Multi-column documents (academic papers, newspapers)
3. Scanned documents with layout complexity
4. Documents with mixed formatting

**When Traditional OCR Wins**:
1. Simple digital PDFs with clean text
2. Speed-critical applications
3. Documents without complex structure
4. Resource-constrained environments

## 6.3 Implications

### 6.3.1 For RAG System Design

1. **Parser Selection Strategy**:
   - Use document classification to route to appropriate parser
   - VLM for complex layouts, OCR for simple documents
   - Hybrid approach optimizes cost-quality trade-off

2. **Chunking Strategy Integration**:
   - VLM output enables structure-aware chunking
   - Consider hierarchical chunking for markdown output
   - Table-as-atomic-unit policy recommended

3. **Quality Assurance**:
   - Monitor parsing quality with CER/WER thresholds
   - Fallback mechanisms for parser failures
   - Human review for high-stakes documents

### 6.3.2 For VLM Prompt Engineering

1. **Transcription vs Extraction**:
   - Transcription-focused prompts reduce hallucination
   - Explicit "do not add" instructions critical
   - [unclear] markers better than guessing

2. **Output Format**:
   - Markdown provides good balance of structure and simplicity
   - JSON/XML for programmatic processing
   - Plain text loses structural benefits

3. **Temperature Settings**:
   - Temperature 0.0 for deterministic, reproducible output
   - Higher temperatures for creative tasks only

### 6.3.3 For Future Research

1. **Dataset Expansion**:
   - More diverse document types needed
   - Larger scale for statistical power
   - Multilingual evaluation

2. **End-to-End Evaluation**:
   - Extend to answer generation quality
   - RAGAs framework integration
   - User satisfaction studies

3. **Efficiency Optimization**:
   - Smaller VLM models (distillation)
   - Batch processing optimization
   - Caching strategies

## 6.4 Limitations

### 6.4.1 Dataset Limitations

1. **Sample Size**: Only 3 documents, 30 Q&A pairs
   - Statistical power limited
   - Results should be considered preliminary
   - Bootstrap CI provides some robustness

2. **Document Diversity**:
   - Limited to Korean and English
   - Specific document types may not generalize
   - No handwritten or highly degraded documents

3. **Ground Truth Quality**:
   - Manual annotation subject to human error
   - Single annotator (no inter-rater reliability)
   - Markdown style choices affect metrics

### 6.4.2 Methodological Limitations

1. **Single VLM Model**:
   - Results may not generalize to other VLMs
   - Qwen3-VL specific behaviors
   - Version and configuration dependencies

2. **Chunking Configuration**:
   - Fixed parameters may not be optimal
   - Structure-agnostic chunking by design
   - Ablation study partially addresses

3. **Embedding Model Choice**:
   - ko-sroberta may have language biases
   - Single embedding model tested
   - Retrieval method (cosine) is basic

### 6.4.3 Practical Limitations

1. **Cost Analysis Incomplete**:
   - GPU costs not quantified
   - Latency measured but not optimized
   - Production deployment considerations missing

2. **Real-World Conditions**:
   - Clean test documents
   - No noisy or low-quality scans
   - Ideal network conditions for VLM API

## 6.5 Threats to Validity

### 6.5.1 Internal Validity

- **Confounding Variables**: Chunking may not be perfectly controlled
- **Measurement Error**: CER/WER sensitive to normalization
- **Selection Bias**: Test documents not randomly sampled

### 6.5.2 External Validity

- **Generalization**: Limited document types
- **Technology Changes**: VLM capabilities evolving rapidly
- **Domain Specificity**: Results may not apply to all domains

### 6.5.3 Construct Validity

- **Metric Relevance**: CER/WER may not fully capture usefulness
- **Retrieval Proxy**: HR@k approximates but != actual utility
- **Ground Truth Definition**: Markdown style choices affect baseline
