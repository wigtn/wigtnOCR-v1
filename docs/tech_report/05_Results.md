# 5. Results

> **Updated**: 2026-01-29
> **Status**: Parsing Test 결과 반영 완료, RQ 재구성 반영

## 5.1 RQ1: 텍스트 추출 품질 전제 검증

> **질문**: OCR/텍스트 추출의 기본 정확도(CER, WER)가 VLM 구조화 단계로 넘어가기에 충분한 수준인가?

CER/WER은 VLM 구조화의 성과 지표가 아니라, 파이프라인의 **전제 조건**을 확인하는 지표이다.

### 5.1.1 Character Error Rate (CER)

| Document | Text-Baseline | Image-Baseline | Text-Advanced | Image-Advanced |
|----------|---------------|----------------|---------------|----------------|
| test_1 (Korean/Scanned) | N/A | 91.87% | N/A | 536.50% |
| test_2 (English/Scanned) | 99.59% | 40.80% | 120.54% | **33.09%** |
| test_3 (English/Digital) | 51.25% | **40.79%** | 64.11% | 57.71% |

### 5.1.2 Word Error Rate (WER)

| Document | Text-Baseline | Image-Baseline | Text-Advanced | Image-Advanced |
|----------|---------------|----------------|---------------|----------------|
| test_1 (Korean/Scanned) | N/A | 99.42% | N/A | 322.63% |
| test_2 (English/Scanned) | 99.69% | 55.59% | 262.94% | **37.31%** |
| test_3 (English/Digital) | 57.19% | **41.24%** | 69.34% | 63.27% |

**Tokenization**:
- English: Whitespace tokenization
- Korean: Whitespace tokenization (MeCab 미적용)

### 5.1.3 전제 조건 판정

| 파서 | CER (test_3) | 전제 충족 여부 | 비고 |
|------|-------------|---------------|------|
| Text-Baseline | 51.25% | △ | 텍스트 추출 가능, 구조 없음 |
| Image-Baseline | 40.79% | ○ | 가장 낮은 CER |
| Text-Advanced | 64.11% | △ | VLM 구조화 시 +13pp CER 증가, 허용 범위 |
| Image-Advanced | 57.71% | △ | VLM 구조화 시 +17pp CER 증가, 허용 범위 |

**판정 기준**: CER < 100%이면 의미있는 텍스트 추출이 이루어진 것으로 판단. 단, test_1의 Image-Advanced (CER 536%)는 hallucination으로 전제 조건 **미충족**.

**핵심 관찰**:
1. **Baseline 파서는 기본 텍스트 추출에 적합**: Image-Baseline이 가장 낮은 CER(40.79%) 달성
2. **Advanced 파서의 CER 증가는 예상된 trade-off**: VLM 구조화 과정에서 텍스트 재구성으로 인한 +13~17pp 증가
3. **한글 스캔 문서에서 Hallucination 위험**: test_1 Image-Advanced CER 536% → VLM 적용 부적합

### 5.1.4 Parser Architecture Comparison

| Parser | Stage 1 | Stage 2 | 적합 문서 유형 |
|--------|---------|---------|----------------|
| Text-Baseline | PyMuPDF | - | 디지털 PDF (텍스트 레이어 존재) |
| Image-Baseline | RapidOCR | - | 스캔 PDF, 이미지 기반 문서 |
| Text-Advanced | PyMuPDF | VLM 구조화 | 디지털 PDF + 구조 추출 필요 시 |
| Image-Advanced | RapidOCR | VLM 구조화 | 스캔 PDF + 구조 추출 필요 시 |

## 5.2 RQ2: 구조 보존 효과

> **질문**: VLM 기반 Two-Stage Parsing을 적용하면 문서의 구조를 얼마나 더 잘 보존하는가?

### 5.2.1 Structure F1 Score

| Document | Text-Baseline | Image-Baseline | Text-Advanced | Image-Advanced |
|----------|---------------|----------------|---------------|----------------|
| test_1 (Korean/Scanned) | N/A | 0.00% | N/A | 0.00% |
| test_2 (English/Scanned) | 0.00% | 0.00% | 9.30% | **16.67%** |
| test_3 (English/Digital) | 0.00% | 0.00% | **79.25%** | 77.78% |

**Key Observations**:
1. **Baseline 파서는 구조 검출 불가**: 모든 Baseline 파서의 Structure F1 = 0%
2. **VLM 구조화 효과 입증**: Advanced 파서에서 Structure F1 크게 향상
3. **디지털 PDF에서 Text-Advanced 최적**: test_3에서 F1 79.25% 달성

### 5.2.2 Precision / Recall 분석 (test_3)

| Parser | Precision | Recall | F1 | TP | FP | FN | Hyp | Ref |
|--------|-----------|--------|----|----|----|-----|-----|-----|
| Text-Baseline | 0.00% | 0.00% | 0.00% | 0 | 11 | 24 | 11 | 24 |
| Image-Baseline | 0.00% | 0.00% | 0.00% | 0 | 0 | 24 | 0 | 24 |
| Text-Advanced | **72.41%** | **87.50%** | **79.25%** | 21 | 8 | 3 | 29 | 24 |
| Image-Advanced | 70.00% | 87.50% | 77.78% | 21 | 9 | 3 | 30 | 24 |

**해석**:
- **Recall 87.5%**: GT의 24개 구조 요소 중 21개 검출 → 누락(FN) 3개만 발생
- **Precision ~71-72%**: 파서 생성 요소 중 약 71%가 GT와 일치 → FP 8-9개 (과잉 생성)
- **FP 원인**: VLM이 일부 텍스트를 구조 요소로 과잉 해석 (hallucination의 일종)
- **FN 원인**: 세부 섹션 헤딩 또는 미세 구조 누락

### 5.2.3 Structure Element Types

평가 대상 구조 요소:

| Element Type | Pattern | Example |
|--------------|---------|---------|
| Heading | `^#{1,6}\s+` | `# Title`, `## Section` |
| Unordered List | `^[\s]*[-*+]\s+` | `- item` |
| Ordered List | `^[\s]*\d+\.\s+` | `1. first` |
| Table Row | `^\|.+\|$` | `\| col1 \| col2 \|` |

## 5.3 RQ3: 다운스트림 효과 (청킹 품질)

> **질문**: 구조 보존이 시맨틱 청킹 품질을 향상시키는가?

### 5.3.1 Chunking Results

- test_2 Image-Advanced: 7 chunks, BC score 0.512 (moderate coherence)
- test_3 Image-Advanced: 18 chunks with natural section boundaries

**Observation**: 마크다운 헤더와 섹션이 자연스러운 청크 경계를 제공하여, 구조화된 파싱 출력이 더 의미있는 청크 분할을 가능하게 한다.

## 5.4 Latency Analysis

### 5.4.1 Processing Time (seconds)

| Parser | test_1 | test_2 | test_3 | 특성 |
|--------|--------|--------|--------|------|
| Text-Baseline | 1.35 | 3.58 | **2.31** | 가장 빠름 (디지털 PDF) |
| Image-Baseline | 18.07 | 23.65 | **0.27** | 스캔 PDF에서 느림 |
| Text-Advanced | 1.83 | 39.01 | 42.92 | VLM 호출 오버헤드 |
| Image-Advanced | 51.26 | 37.06 | 35.75 | OCR + VLM 이중 처리 |

### 5.4.2 Stage별 시간 분석 (test_3)

| Parser | Stage 1 | Stage 2 (VLM) | Total |
|--------|---------|---------------|-------|
| Text-Advanced | 2.28s | 40.64s | 42.92s |
| Image-Advanced | 0.27s | 35.48s | 35.75s |

**Observation**: VLM 구조화 단계가 전체 시간의 90% 이상 차지

## 5.5 Key Findings Summary

### 5.5.1 Research Question Answers

| RQ | 질문 | 판정 | 근거 |
|----|------|------|------|
| **RQ1** | OCR 추출 품질이 VLM 입력으로 충분한가? | **조건부 Yes** | 영어 문서 CER 40-51%로 충분. 한글 스캔 문서는 hallucination 위험 |
| **RQ2** | VLM Two-Stage Parsing이 구조를 더 잘 보존하는가? | **Yes** | Structure F1: 0% → 79.25% (test_3). Recall 87.5% |
| **RQ3** | 구조 보존이 청킹 품질을 향상시키는가? | **간접 Yes** | BC 0.512, 18개 자연 섹션 경계 생성 |

### 5.5.2 Trade-off 분석

**왜 이 비교가 필요한가?**: 실무에서는 정확도와 구조화 품질 중 하나만 선택해야 하는 상황이 발생한다. VLM 구조화는 Structure F1을 +79%p 올리지만, CER은 +17%p 증가하고 처리 시간은 159배 느려진다. 어떤 시나리오에서 어떤 전략을 선택해야 하는가?

**정량적 비교** (test_3 기준):

| 지표 | Baseline (최선) | Advanced (최선) | 차이 |
|------|----------------|----------------|------|
| CER | 40.79% (Image) | 57.71% (Image) | +16.92pp |
| Structure F1 | 0% | 79.25% (Text) | +79.25pp |
| Latency | 0.27s (Image) | 42.92s (Text) | ×159 |

```
                    텍스트 정확도 (CER ↓)
                           ▲
                           │
         Baseline          │         (이상적)
         ┌─────┐           │
         │ 좋음 │           │
         └─────┘           │
                           │
    ─────────────────────────────────────▶ 구조화 품질 (F1 ↑)
                           │
                           │         Advanced
                           │         ┌─────┐
                           │         │ 좋음 │
                           │         └─────┘
```

**시나리오별 추천**:

| 시나리오 | 권장 파서 | 이유 |
|----------|----------|------|
| 속도 우선 (실시간 검색) | Baseline | Latency 0.27-2.3s, CER 최선 |
| 구조 우선 (RAG 청킹) | Advanced | Structure F1 79%, 청크 품질 향상 |
| 하이브리드 (문서 분류 후 라우팅) | 혼합 | 복잡 문서→Advanced, 단순 문서→Baseline |

### 5.5.3 Parser Selection Guide

| 사용 목적 | 권장 파서 | 이유 |
|----------|----------|------|
| 텍스트 검색/인덱싱 | Baseline | 높은 텍스트 정확도 |
| RAG/Chunking | Advanced | 구조 기반 청킹 가능 |
| 문서 변환 (HTML/MD) | Advanced | 마크다운 구조 활용 |
| 실시간 처리 | Baseline | 낮은 Latency |

---

## Appendix: Raw Data

### A. test_1 (Korean Scanned PDF)
```
Document: 공공AX 프로젝트 공모안내서
Pages: 5
Type: Scanned PDF

Text-Baseline:  Extraction failed (scanned PDF)
Image-Baseline: CER=91.87%, WER=99.42%, F1=0.00%, Latency=18.07s
Text-Advanced:  Extraction failed (scanned PDF)
Image-Advanced: CER=536.50%, WER=322.63%, F1=0.00%, Latency=51.26s
```

### B. test_2 (English Scanned PDF)
```
Document: Chain-of-Thought Prompting (NeurIPS 2022)
Pages: 4
Type: Scanned PDF

Text-Baseline:  CER=99.59%, WER=99.69%, F1=0.00%, Latency=3.58s
Image-Baseline: CER=40.80%, WER=55.59%, F1=0.00%, Latency=23.65s
Text-Advanced:  CER=120.54%, WER=262.94%, F1=9.30%, Latency=39.01s
Image-Advanced: CER=33.09%, WER=37.31%, F1=16.67%, Latency=37.06s
```

### C. test_3 (English Digital PDF)
```
Document: Attention Is All You Need (NeurIPS 2017)
Pages: 15
Type: Digital PDF

Text-Baseline:  CER=51.25%, WER=57.19%, F1=0.00%, Latency=2.31s
Image-Baseline: CER=40.79%, WER=41.24%, F1=0.00%, Latency=0.27s
Text-Advanced:  CER=64.11%, WER=69.34%, F1=79.25%, Latency=42.92s
Image-Advanced: CER=57.71%, WER=63.27%, F1=77.78%, Latency=35.75s
```
