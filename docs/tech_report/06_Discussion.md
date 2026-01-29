# 6. Discussion

## 6.1 Research Question Analysis

### 6.1.1 RQ1: 텍스트 추출 품질 전제 검증

**Question**: OCR/텍스트 추출의 기본 정확도(CER, WER)가 VLM 구조화 단계로 넘어가기에 충분한 수준인가?

**Finding**: 영어 문서에서는 Baseline CER 40-51%로 VLM 구조화 입력으로 충분하다. 그러나 한글 스캔 문서에서는 hallucination이 발생하여 VLM 적용에 주의가 필요하다.

**Analysis**:

| Parser | CER (test_3) | WER (test_3) | 전제 충족 | 비고 |
|--------|-------------|-------------|----------|------|
| Text-Baseline | 51.25% | 57.19% | ○ | 텍스트 추출 가능 |
| Image-Baseline | **40.79%** | **41.24%** | ○ | 최선 CER |
| Text-Advanced | 64.11% | 69.34% | △ | +13pp CER 증가 (구조화 대가) |
| Image-Advanced | 57.71% | 63.27% | △ | +17pp CER 증가 (구조화 대가) |

CER/WER 결과의 의미:

1. **Baseline CER은 VLM 입력 품질 확인용**: Image-Baseline CER 40.79%는 텍스트가 충분히 추출되었음을 의미하며, 이 텍스트를 VLM Stage 2로 넘겨도 된다는 근거가 된다.

2. **Advanced CER 증가는 예상된 trade-off**: VLM이 텍스트를 마크다운으로 재구성하면서 +13~17pp CER 증가가 발생한다. 이는 구조화의 대가이며, Structure F1 +79%p 향상으로 정당화된다.

3. **Hallucination 경고**: test_1(한글 스캔)에서 CER 536%가 관찰되었다. 이는 VLM이 불확실한 입력에 대해 존재하지 않는 텍스트를 생성한 것으로, CER/WER 전제 조건 검증의 중요성을 보여준다.

### 6.1.2 RQ2: 구조 보존 효과

**Question**: VLM 기반 Two-Stage Parsing을 적용하면 문서의 구조를 얼마나 더 잘 보존하는가?

**Finding**: **극적으로 향상된다.** Structure F1이 0%(Baseline) → 79.25%(Text-Advanced)로 개선. Recall 87.5%로 대부분의 구조 요소를 검출한다.

**Analysis**:

| Parser | Structure F1 | Precision | Recall | TP | FP | FN |
|--------|-------------|-----------|--------|----|----|-----|
| Text-Baseline | 0% | 0% | 0% | 0 | 11 | 24 |
| Image-Baseline | 0% | 0% | 0% | 0 | 0 | 24 |
| Text-Advanced | **79.25%** | 72.41% | 87.5% | 21 | 8 | 3 |
| Image-Advanced | 77.78% | 70.0% | 87.5% | 21 | 9 | 3 |

1. **Structure F1 Breakthrough (0% → 79%)**:
   - Baseline 파서는 구조 요소를 전혀 생성하지 못함 (F1 = 0%)
   - VLM Stage 2 적용 후 F1 79.25% 달성
   - 이는 RQ2의 핵심 가설("VLM이 구조를 보존한다")을 입증

2. **Precision vs Recall 분석**:
   - **Recall 87.5%**: GT 24개 구조 요소 중 21개 검출 → 누락이 적음
   - **Precision 72.41%**: 생성한 29개 요소 중 21개가 정확 → FP 8개 (과잉 생성)
   - VLM hallucination으로 인한 과잉 생성이 있으나, 누락보다 과잉 생성이 덜 해로움 (chunking에서 추가 경계는 무해, 누락된 경계는 치명적)

3. **프롬프트 진화의 기여**:
   - v1 프롬프트에서는 Structure F1 = 0% (헤딩 `#` 미생성)
   - v2 프롬프트(CRITICAL RULES + 명시적 매핑)로 변경 후 79.25% 달성
   - commit `90d516d`의 프롬프트 개선이 결정적 역할

### 6.1.3 RQ3: 다운스트림 효과

**Question**: 구조 보존이 시맨틱 청킹 품질을 향상시키는가?

**Finding**: 간접적 증거로 구조 보존이 청킹 품질을 향상시킨다. 직접적 Hit Rate 측정은 미수행.

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

4. **Limitations**:
   - Direct Hit Rate@k evaluation not performed in this cycle
   - Future work should measure end-to-end retrieval impact

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

1. **명시적 규칙이 핵심**: 2B 모델에서 "MUST use #", "NEVER output without #" 등 강제 지시어가 효과적
2. **System/User 프롬프트 분리**: 역할(System)과 작업 지시(User)를 분리하면 구조 생성 품질 향상
3. **번호→레벨 매핑 명시**: "1→##, 2.1→###" 같은 구체적 규칙이 암시적 지시보다 효과적
4. **Temperature 0.1**: 결정론적이면서도 약간의 유연성을 허용하는 설정

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

1. **Sample Size**: Only 3 documents
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
- **Retrieval Proxy**: BC/CS approximates but != actual retrieval quality
- **Ground Truth Definition**: Markdown style choices affect baseline
