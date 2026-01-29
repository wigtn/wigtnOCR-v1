# 7. Conclusion

## 7.1 Summary of Findings

This study investigated the impact of Vision-Language Model (VLM) based document parsing on structural preservation and semantic chunking quality in RAG systems. Our key findings are:

### 7.1.1 RQ1: 텍스트 추출 품질 전제 검증

- Baseline 파서의 CER 40-51%로 VLM 구조화 입력으로 **충분한 수준**
- Advanced 파서의 CER은 +13~17pp 증가하나, 구조화의 예상된 trade-off로 **허용 범위**
- **한글 스캔 문서 예외**: CER 536% hallucination 발생 → VLM 적용 부적합

### 7.1.2 RQ2: 구조 보존 효과 (핵심 결과)

- **Structure F1: 0% → 79.25%** (Text-Advanced), 극적 개선
- Recall: 87.5% (21/24 structural elements detected), Precision: 72.41%
- True Positives: 21, False Positives: 8, False Negatives: 3
- 프롬프트 v1→v2 진화(commit `90d516d`)가 0% → 79% 개선의 핵심 요인

### 7.1.3 RQ3: 다운스트림 효과

- Boundary Coherence (BC) score of 0.512 achieved with 18 natural chunk divisions
- 마크다운 구조가 자연스러운 청크 경계를 제공
- Direct retrieval metrics (Hit Rate, MRR) 미측정 → Future work

## 7.2 Core Contributions

1. **Multi-Metric Evaluation Framework**: CER/WER(전제 검증) → Structure F1(구조 보존) → BC/CS(다운스트림 효과)의 3단계 평가 체계

2. **Empirical Validation**: Structure F1 0% → 79.25% 정량적 입증

3. **Prompt Engineering Insights**: 2B 소형 모델에서 명시적 규칙("MUST", "NEVER", 번호→레벨 매핑)이 암시적 지시보다 효과적임을 검증

4. **Error Taxonomy**: Hallucination(CER 536%), 과잉 생성(FP 8-9), 누락(FN 3) 등 오류 유형 분류

5. **Hybrid Strategy Recommendation**: 문서 특성에 따른 파서 선택 가이드라인

## 7.3 Hybrid Parsing Strategy

Based on our findings, we recommend the following decision tree for document parsing:

```
                    Document Input
                         │
                         ▼
                    ┌─────────┐
                    │ Scanned?│──────Yes────► VLM (Required)
                    └────┬────┘              ⚠ 한글은 주의
                         │No
                         ▼
                    ┌──────────────┐
                    │ Complex      │
                    │ Layout?      │──────Yes────► VLM (Recommended)
                    │ (Tables,     │
                    │ Multi-column)│
                    └──────┬───────┘
                           │No
                           ▼
                    PyMuPDF (Fast, Sufficient)
```

### 7.3.1 Document Routing Criteria

| Characteristic | Route to VLM | Route to OCR |
|----------------|--------------|--------------|
| Scanned/Image | Always | Never |
| Tables present | Yes | Simple only |
| Multi-column | Yes | No |
| Headers important | Yes | Optional |
| Speed critical | No | Yes |
| Cost sensitive | No | Yes |

### 7.3.2 Expected Outcomes

| Strategy | Structure F1 | CER | Latency | Best For |
|----------|-------------|-----|---------|----------|
| Advanced (VLM) | ~79% | ~58-64% | 35-43s | Structure-critical docs |
| Baseline (OCR) | 0% | ~41-51% | 0.3-2.3s | Speed-critical, simple docs |
| Hybrid | Variable | Variable | Variable | Balanced quality/speed |

**Quantified Trade-offs** (test_3 benchmark):
- Structure F1 gain: +79 percentage points (0% → 79%)
- CER cost: +17 percentage points (41% → 58%)
- Latency cost: 159x slower (0.27s → 42.92s)

## 7.4 Practical Recommendations

### 7.4.1 For RAG System Developers

1. **Implement Document Classification**: Route documents to appropriate parser based on complexity
2. **Use Explicit Rule Prompts**: "MUST", "NEVER" 지시어와 번호→레벨 매핑을 프롬프트에 포함
3. **Enable Structure-Aware Chunking**: Leverage markdown structure for better chunk boundaries
4. **Monitor CER/WER as Prerequisite**: VLM 적용 전 Stage 1 추출 품질 확인

### 7.4.2 For Researchers

1. **Expand Evaluation Datasets**: Include more document types and languages
2. **Study End-to-End Impact**: Measure answer quality, not just retrieval
3. **Compare VLM Models**: Evaluate across different VLM architectures
4. **Investigate Efficiency**: Smaller models, quantization, batching

## 7.5 Future Work

### 7.5.1 Short-Term

1. **End-to-End RAG Evaluation**: Hit Rate@k, MRR 직접 측정
2. **한글 문서 Hallucination 완화**: 한국어 특화 프롬프트, 입력 품질 필터링
3. **Ablation Completion**: 프롬프트 버전별 비교 실험 완료

### 7.5.2 Medium-Term

1. **Hybrid System Implementation**: 자동 문서 분류기 + 파서 라우팅
2. **Multilingual Expansion**: 중국어, 일본어 문서 지원
3. **Efficiency Research**: 모델 경량화, 배치 처리 최적화

### 7.5.3 Long-Term

1. **Adaptive Parsing**: 문서 유형별 최적 파서 자동 학습
2. **Benchmark Publication**: 공개 벤치마크 데이터셋 구축

## 7.6 Closing Remarks

This study provides quantitative evidence that **structural preservation matters** in document parsing for RAG systems. The key insight is:

1. CER/WER은 파싱 품질의 전제 조건이지, VLM의 목표 지표가 아니다
2. VLM의 핵심 기여는 Structure F1 0% → 79%의 구조화 효과이다
3. 소형 모델(2B)에서도 명시적 프롬프트 규칙으로 충분한 구조화가 가능하다

We advocate for a **hybrid approach** that balances quality and efficiency, guided by the three-stage evaluation framework: CER/WER → Structure F1 → BC/CS.

---

**Data Availability**

The ground truth datasets, evaluation code, and experiment configurations are available at:
- Repository: [To be published]
- License: [To be determined]
