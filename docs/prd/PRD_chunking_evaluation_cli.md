# Chunking Evaluation CLI PRD

> **Version**: 2.0
> **Created**: 2026-01-26
> **Updated**: 2026-01-26 (메트릭 정리: BC/CS only)
> **Status**: Draft

## 1. Overview

### 1.1 Problem Statement

현재 VLM Document Parsing 프레임워크는 Phase 1 (Lexical Accuracy: CER, WER) CLI 테스트가 완료되었으나, Phase 2 (Structural Integrity) 평가 시스템이 CLI로 통합되지 않았다.

**현재 상황**:
- `src/test_parsers.py`: CER, WER, Latency 측정 CLI 완성
- `src/chunking/chunker.py`: Chunking 전략 구현 완료 (Fixed, Recursive, Semantic, Hierarchical)
- `src/chunking/metrics.py`: 기존 메트릭 구현 (교체 필요)

**문제점**:
1. 파싱 결과를 Chunking하고 저장하는 파이프라인 없음
2. 청킹 품질을 **직접** 정량화할 수 있는 CLI 없음
3. 파서별 Chunking 품질 비교가 불가능
4. 중간 결과물(Chunk JSON)을 저장/재사용할 수 없음

### 1.2 Goals

- CLI 기반 Chunking + 구조적 평가 시스템 구축
- **MoC 논문의 BC/CS 메트릭으로 청킹 품질 직접 정량화**
- Chunk 데이터 저장 및 재사용 가능한 구조
- Ground Truth 없이도 청킹 품질 평가 가능

### 1.3 Non-Goals (Out of Scope)

- Phase 3 Retrieval 평가 (다음 단계)
- Streamlit UI 통합 (별도 작업)
- 실시간 비교 대시보드
- 라벨 기반 평가 메트릭 (Pk, WindowDiff 등)

### 1.4 Scope

| 포함 | 제외 |
|------|------|
| CLI 기반 Chunking 테스트 | Web UI 통합 |
| BC, CS 메트릭 계산 | 라벨 기반 평가 |
| Chunk 데이터 JSON 저장 | 임베딩 벡터 저장 |
| 파서별 Chunking 비교 | Q&A 데이터셋 생성 |
| **Ground Truth 불필요 평가** | 위치 매칭 기반 BS |

## 2. Metric Philosophy

> **Reference**: MoC Paper (arXiv:2503.09600v2) - "Mixtures of Text Chunking Learners for RAG System"

### 2.1 왜 BC/CS인가?

**기존 라벨 기반 평가의 한계**:
- Ground Truth 경계 라벨 필요 → 도메인 전문가 필요, 비용 높음
- 문서 업데이트 시 재라벨링 필요 → 프로덕션에서 **지속 불가능**
- 한 번은 가능해도 반복 측정 불가

**BC/CS의 장점**:
- **라벨 불필요**: Ground Truth 없이 계산 가능
- **반복 측정 가능**: 청킹 전략 변경, 문서 업데이트 시 자동 재계산
- **RAG 목적 직접 연결**: Retrieval 성능 이전 단계에서 "청크 자체가 말이 되는가?" 평가
- **모델 인식 기준**: RAG의 최종 소비자는 LLM → 사람 직관 아닌 **모델 인식 구조** 기준

**논문 검증 결과** (Pearson 상관계수):
| Metric | ROUGE-L 상관 | 해석 |
|--------|-------------|------|
| BC | **0.8776** | 강한 양의 상관 |
| CS_complete | **-0.7453** | 강한 음의 상관 |
| CS_incomplete | **-0.6663** | 중간 음의 상관 |

### 2.2 메트릭 정의

#### Boundary Clarity (BC) - 경계 명확성

청크가 의미 단위를 얼마나 효과적으로 분리하는지 측정:

```
BC(q, d) = ppl(q|d) / ppl(q)
```

- `ppl(q)`: 문장 시퀀스 q의 perplexity
- `ppl(q|d)`: 텍스트 청크 d가 주어졌을 때의 contrastive perplexity
- **값 범위**: [0, 1]
- **해석**:
  - 1에 가까움: 두 청크가 의미적으로 독립적 (좋음)
  - 0에 가까움: 강한 의미적 상호의존성 (나쁨)
- **높을수록 좋음**

#### Chunk Stickiness (CS) - 청크 점착성

청크 간 의미 연관 그래프의 Structural Entropy로 측정:

```
Edge(q, d) = (ppl(q) - ppl(q|d)) / ppl(q)   # 범위: [0, 1]

CS(G) = -Σ (hi/2m) · log2(hi/2m)
```

- `G`: 의미 연관 그래프
- `hi`: 노드 i의 degree
- `m`: 총 edge 수
- **낮을수록 좋음**: 청크 내부는 응집력 있고, 청크 간에는 독립적

#### 그래프 구성 방식

1. **Complete Graph**: 모든 청크 쌍의 Edge 계산 후 threshold K로 필터링
2. **Incomplete Graph (Sequential)**: 순차적 위치 제약 추가 (효율성 향상)
   - Relevance Criterion: Edge(di, dj) > K
   - Sequential Constraint: j - i > δ (δ=0 권장)

### 2.3 핵심 관점

| 구분 | 라벨 기반 (기존) | BC/CS (MoC) |
|------|---------------|-------------|
| 질문 | "정답과 얼마나 비슷한가?" | "모델 입장에서 자연스러운가?" |
| 의존성 | Ground Truth 필수 | 불필요 |
| 재현성 | 낮음 | 높음 |
| 프로덕션 적합 | ❌ | ✅ |
| 목적 | 절대 점수 | **비교 가능한 점수** |

## 3. User Stories

### 3.1 Primary User

As a **ML/NLP 연구자**, I want to **파싱된 문서의 청킹 품질을 Ground Truth 없이 평가**하여 so that **VLM 파서가 OCR 대비 더 나은 Chunk를 생성하는지 운영 환경에서 반복 검증**할 수 있다.

### 3.2 Acceptance Criteria (Gherkin)

**Scenario 1: 단일 문서 Chunking 품질 평가**
```gherkin
Scenario: 파싱 결과를 Chunking하고 BC/CS로 평가
  Given PDF 파일이 존재 (Ground Truth 불필요)
  When `python -m src.test_chunking --input data/test_1/test_data_1.pdf` 실행
  Then 각 파서별(VLM, OCR-Text, OCR-Image) Chunking 수행
  And BC, CS 점수가 콘솔에 출력
  And Chunk 데이터가 `results/chunks/` 폴더에 JSON으로 저장
```

**Scenario 2: 기존 파싱 결과로 Chunking**
```gherkin
Scenario: 이미 파싱된 결과 파일로 Chunking 평가
  Given 이전에 저장된 파싱 결과 파일 (vlm_output.md, ocr_output.txt)
  When `python -m src.test_chunking --parsed-dir results/parsing/20260126/` 실행
  Then 파싱 단계 스킵하고 Chunking부터 수행
  And BC, CS 점수 출력
```

**Scenario 3: Chunking 전략 비교**
```gherkin
Scenario: 여러 Chunking 전략 비교
  Given 파싱된 텍스트
  When `python -m src.test_chunking --strategies recursive,semantic,hierarchical` 실행
  Then 각 전략별 BC, CS 점수 비교 테이블 출력
```

## 4. Functional Requirements

| ID | Requirement | Priority | Dependencies |
|----|------------|----------|--------------|
| FR-001 | CLI 진입점 `src/test_chunking.py` 생성 | P0 (Must) | - |
| FR-002 | PDF/이미지 → 파싱 → Chunking 파이프라인 | P0 (Must) | test_parsers.py |
| FR-003 | Boundary Clarity (BC) 계산 및 출력 | P0 (Must) | LLM API |
| FR-004 | Chunk Stickiness (CS) 계산 및 출력 | P0 (Must) | LLM API |
| FR-005 | Chunk 데이터 JSON 저장 | P0 (Must) | - |
| FR-006 | 기존 파싱 결과 파일 입력 지원 | P1 (Should) | - |
| FR-007 | 다중 Chunking 전략 비교 | P1 (Should) | chunker.py |
| FR-008 | 결과 요약 Markdown 생성 | P1 (Should) | - |
| FR-009 | `--skip-vlm` 등 기존 옵션 호환 | P1 (Should) | test_parsers.py |
| FR-010 | 파서별 상세 메트릭 JSON 저장 | P2 (Could) | - |
| FR-011 | Complete/Incomplete Graph 선택 옵션 | P1 (Should) | - |
| FR-012 | Threshold K 파라미터 | P1 (Should) | 기본: 0.8 |

## 5. Non-Functional Requirements

### 5.1 Performance
- Chunking 처리: < 1초 (500 chunks 기준)
- BC/CS 계산: LLM API 호출 시간에 의존 (배치 처리로 최적화)
- 메모리 사용: < 2GB

### 5.2 Usability
- `test_parsers.py`와 동일한 CLI 패턴
- 명확한 진행 상황 출력 (프로그레스 바 또는 단계 표시)
- 에러 메시지에 해결 방법 포함

### 5.3 Compatibility
- Python 3.11+
- LLM API 필요 (Qwen, GPT 등)
- 기존 `chunker.py` 모듈 활용

## 6. Technical Design

### 6.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      test_chunking.py (New)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │   Input     │     │   Parse     │     │   Chunk     │          │
│   │  (PDF/MD)   │ ──► │ (Optional)  │ ──► │  (Chunker)  │          │
│   └─────────────┘     └─────────────┘     └─────────────┘          │
│                                                  │                   │
│                                                  ▼                   │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │   Output    │ ◄── │   Report    │ ◄── │  Evaluate   │          │
│   │  (JSON/MD)  │     │  (Summary)  │     │  (BC, CS)   │          │
│   └─────────────┘     └─────────────┘     └─────────────┘          │
│                                                  │                   │
│                                                  ▼                   │
│                                          ┌─────────────┐            │
│                                          │  LLM API    │            │
│                                          │ (Perplexity)│            │
│                                          └─────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 CLI Interface

```bash
# 기본 사용법 (Ground Truth 불필요!)
python -m src.test_chunking \
    --input data/test_1/test_data_1.pdf

# 전체 옵션
python -m src.test_chunking \
    --input data/test_1/test_data_1.pdf \
    --strategy recursive_character \       # Chunking 전략
    --chunk-size 500 \                     # Chunk 크기
    --chunk-overlap 50 \                   # Overlap
    --output-dir results/chunks/ \         # 출력 디렉토리
    --skip-vlm \                           # VLM 스킵
    --graph-type incomplete \              # Graph 유형 (complete/incomplete)
    --threshold-k 0.8 \                    # Edge 필터링 threshold
    --llm-model qwen2.5-7b \              # Perplexity 계산용 LLM
    --verbose                              # 상세 출력

# 기존 파싱 결과로 Chunking
python -m src.test_chunking \
    --parsed-files vlm_output.md,ocr_output.txt

# 다중 전략 비교
python -m src.test_chunking \
    --input data/test_1/test_data_1.pdf \
    --strategies recursive,semantic,hierarchical
```

### 6.3 Output Format

#### Console Output
```
============================================================
🔪 VLM Document Chunking Quality Test
============================================================
📄 입력 파일: data/test_1/test_data_1.pdf
⚙️ Chunking 전략: recursive_character (500/50)
🤖 LLM: qwen2.5-7b (perplexity 계산용)

------------------------------------------------------------
📊 Phase 1: Parsing
------------------------------------------------------------
   VLM: ✓ (2.34s, 1523 chars)
   OCR-Text: ✓ (0.12s, 1456 chars)
   OCR-Image: ✓ (3.21s, 1389 chars)

------------------------------------------------------------
📊 Phase 2: Chunking
------------------------------------------------------------
   VLM: 5 chunks (avg 304 chars)
   OCR-Text: 6 chunks (avg 242 chars)
   OCR-Image: 7 chunks (avg 198 chars)

------------------------------------------------------------
📊 Phase 3: Quality Evaluation (BC/CS)
------------------------------------------------------------

| Parser    | Chunks | BC (↑) | CS_c (↓) | CS_i (↓) |
|-----------|--------|--------|----------|----------|
| VLM       | 5      | 0.864  | 2.125    | 1.438    |
| OCR-Text  | 6      | 0.845  | 2.250    | 1.483    |
| OCR-Image | 7      | 0.814  | 2.325    | 1.650    |

🏆 Best BC: VLM (0.864) - 청크 경계가 가장 명확
🏆 Best CS: VLM (1.438) - 청크 간 독립성이 가장 높음

------------------------------------------------------------
💾 결과 저장
------------------------------------------------------------
   ✓ results/chunks/vlm_chunks.json (5 chunks)
   ✓ results/chunks/ocr-text_chunks.json (6 chunks)
   ✓ results/chunks/evaluation.json
   ✓ results/chunks/README.md (요약)
```

#### Evaluation JSON (`evaluation.json`)
```json
{
  "metadata": {
    "source_file": "test_data_1.pdf",
    "chunking_config": {
      "strategy": "recursive_character",
      "chunk_size": 500,
      "chunk_overlap": 50
    },
    "evaluation_config": {
      "llm_model": "qwen2.5-7b",
      "graph_type": "incomplete",
      "threshold_k": 0.8
    },
    "timestamp": "2026-01-26T10:30:00Z"
  },
  "results": {
    "VLM": {
      "chunk_count": 5,
      "boundary_clarity": 0.864,
      "chunk_stickiness_complete": 2.125,
      "chunk_stickiness_incomplete": 1.438
    },
    "OCR-Text": { ... },
    "OCR-Image": { ... }
  },
  "comparison": {
    "best_bc": "VLM",
    "best_cs": "VLM"
  }
}
```

### 6.4 BC/CS 계산 구현

```python
# src/chunking/metrics.py (새로 구현)

def calculate_perplexity(text: str, llm) -> float:
    """텍스트의 perplexity 계산"""
    # LLM API를 사용하여 log probability 계산
    pass

def calculate_contrastive_perplexity(query: str, context: str, llm) -> float:
    """컨텍스트가 주어졌을 때의 perplexity 계산"""
    # ppl(q|d)
    pass

def boundary_clarity(chunk_q: str, chunk_d: str, llm) -> float:
    """
    BC(q, d) = ppl(q|d) / ppl(q)

    Returns: 0~1, 높을수록 좋음 (두 청크가 독립적)
    """
    ppl_q = calculate_perplexity(chunk_q, llm)
    ppl_q_given_d = calculate_contrastive_perplexity(chunk_q, chunk_d, llm)
    return ppl_q_given_d / ppl_q

def edge_weight(chunk_q: str, chunk_d: str, llm) -> float:
    """
    Edge(q, d) = (ppl(q) - ppl(q|d)) / ppl(q)

    Returns: 0~1, 1에 가까우면 높은 상관관계
    """
    ppl_q = calculate_perplexity(chunk_q, llm)
    ppl_q_given_d = calculate_contrastive_perplexity(chunk_q, chunk_d, llm)
    return (ppl_q - ppl_q_given_d) / ppl_q

def chunk_stickiness(chunks: list, llm, threshold_k: float = 0.8,
                     graph_type: str = "incomplete") -> float:
    """
    CS(G) = -Σ (hi/2m) · log2(hi/2m)  (Structural Entropy)

    Returns: 낮을수록 좋음 (청크 간 독립적)
    """
    import math

    # 1. 그래프 구성
    n = len(chunks)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            # Incomplete graph: sequential constraint
            if graph_type == "incomplete" and j - i > 1:
                continue

            weight = edge_weight(chunks[j], chunks[i], llm)
            if weight > threshold_k:
                edges.append((i, j, weight))

    # 2. Structural Entropy 계산
    m = len(edges)
    if m == 0:
        return 0.0

    # 각 노드의 degree 계산
    degrees = [0] * n
    for i, j, _ in edges:
        degrees[i] += 1
        degrees[j] += 1

    entropy = 0.0
    for h_i in degrees:
        if h_i > 0:
            p = h_i / (2 * m)
            entropy -= p * math.log2(p)

    return entropy
```

### 6.5 File Structure

```
results/
├── chunks/
│   ├── {timestamp}/
│   │   ├── vlm_chunks.json
│   │   ├── ocr-text_chunks.json
│   │   ├── ocr-image_chunks.json
│   │   ├── evaluation.json
│   │   └── README.md
```

## 7. Implementation Phases

### Phase 1: Core CLI + BC 계산 (MVP)
- [ ] `src/test_chunking.py` 신규 생성
- [ ] 기본 CLI 인터페이스 (--input, --output-dir)
- [ ] 파싱 로직 연동 (test_parsers.py 함수 import)
- [ ] Chunking 파이프라인 구현 (chunker.py 호출)
- [ ] Perplexity 계산 로직 구현 (LLM API 연동)
- [ ] BC (Boundary Clarity) 계산 및 출력
- [ ] Chunk 데이터 JSON 저장

**Deliverable**: `python -m src.test_chunking --input PDF` 실행 가능

### Phase 2: CS 계산 + 그래프 구성
- [ ] Edge weight 계산 로직 구현
- [ ] Complete/Incomplete Graph 구성
- [ ] CS (Chunk Stickiness) 계산 - Structural Entropy
- [ ] `--graph-type`, `--threshold-k` 옵션 추가
- [ ] evaluation.json 저장 구현
- [ ] 파서별 비교 테이블 출력

**Deliverable**: BC + CS 완전 평가 시스템

### Phase 3: Advanced Features
- [ ] 기존 파싱 결과 파일 입력 (`--parsed-files`, `--parsed-dir`)
- [ ] 다중 Chunking 전략 비교 (`--strategies`)
- [ ] test_parsers.py 옵션 호환 (--skip-vlm, --skip-docling)
- [ ] 결과 요약 Markdown 생성
- [ ] 에러 핸들링 및 사용자 친화적 메시지
- [ ] pyproject.toml 스크립트 등록

**Deliverable**: 완전한 CLI 도구

### Phase 4: Documentation & Testing
- [ ] README.md 업데이트 (사용법 추가)
- [ ] 단위 테스트 작성 (tests/test_chunking_cli.py)
- [ ] 벤치마크 테스트 실행 (test_1, test_2, test_3)
- [ ] 코드 정리 및 타입 힌트 보완

**Deliverable**: 문서화 및 테스트 완료

## 8. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| CLI 실행 성공률 | 100% | 모든 테스트 케이스 통과 |
| BC/CS 계산 일관성 | 논문 결과와 유사한 범위 | Table 3 참고 |
| 처리 시간 (Chunking만) | < 1초 | 벤치마크 |
| 문서화 | README 업데이트 | 사용법 가이드 |

**논문 벤치마크 참고값** (Qwen2.5-7B):
| Method | BC (↑) | CS_c (↓) | CS_i (↓) |
|--------|--------|----------|----------|
| Fixed | 0.8049 | 2.421 | 1.898 |
| Llama_index | 0.8455 | 2.250 | 1.483 |
| Semantic | 0.8140 | 2.325 | 1.650 |
| LLM (Qwen2.5-14B) | 0.8641 | 2.125 | 1.438 |

## 9. Dependencies

### 9.1 Internal Modules
- `src/chunking/chunker.py`
- `src/chunking/metrics.py` (새로 구현)
- `src/test_parsers.py` (파싱 로직 재사용)
- `src/parsers/vlm_parser.py`
- `src/parsers/ocr_parser.py`

### 9.2 External Libraries
- LLM API (Qwen, GPT 등) - Perplexity 계산용
- `numpy` - Entropy 계산
- `networkx` (선택) - 그래프 구성

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM API 비용 | BC/CS 계산 비용 증가 | 로컬 LLM 지원, 캐싱 |
| Perplexity 계산 시간 | 전체 평가 시간 증가 | 배치 처리, 병렬화 |
| LLM 선택에 따른 결과 차이 | 일관성 문제 | 동일 LLM 사용 권장, 모델명 기록 |

## 11. Open Questions

1. **LLM 선택**: 로컬 Qwen vs API (GPT)?
2. **캐싱 전략**: Perplexity 결과 캐싱 여부?
3. **배치 크기**: Perplexity 계산 시 최적 배치 크기?

---

## References

- Zhao et al. (2025). "MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System" (arXiv:2503.09600v2)
- Table 3: Performance of different chunking methods under various LMs
- Table 9: Pearson correlation coefficients (BC↔ROUGE-L: 0.8776)

## Next Steps

1. PRD 검토 완료 ✅
2. 구현 시작 (`/implement chunking-evaluation-cli`)
3. 테스트 및 문서화
