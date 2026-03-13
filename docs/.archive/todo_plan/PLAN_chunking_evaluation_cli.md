# Task Plan: Chunking Evaluation CLI

> **Generated from**: docs/prd/PRD_chunking_evaluation_cli.md
> **Created**: 2026-01-26
> **Updated**: 2026-01-26 (메트릭 정리: BC/CS only)
> **Status**: ready
> **Reference**: MoC Paper (arXiv:2503.09600v2)

## Execution Config

| Option | Value | Description |
|--------|-------|-------------|
| `auto_commit` | true | 완료 시 자동 커밋 |
| `commit_per_phase` | true | Phase별 중간 커밋 |
| `quality_gate` | true | /auto-commit 품질 검사 |

## Metric Decision Summary

### 왜 BC/CS만 사용하는가?

**기존 라벨 기반 평가 (BS 등)의 한계**:
- Ground Truth 필수 → 프로덕션에서 **지속 불가능**
- 도메인 전문가 필요, 비용 높음
- 문서 업데이트마다 재라벨링 필요

**BC/CS의 장점** (MoC 논문):
- **라벨 불필요**: Ground Truth 없이 계산 가능
- **반복 측정 가능**: 운영 중 자동 재계산
- **RAG 성능과 강한 상관관계**: Pearson 0.88 (BC), -0.75 (CS)
- **모델 인식 기준**: RAG 소비자는 LLM → 모델 관점 평가

### 메트릭 정의

| 메트릭 | 공식 | 해석 |
|--------|------|------|
| **BC** (Boundary Clarity) | `ppl(q\|d) / ppl(q)` | 높을수록 좋음 (청크 경계 명확) |
| **CS** (Chunk Stickiness) | Structural Entropy | 낮을수록 좋음 (청크 간 독립적) |

## Phases

### Phase 1: Core CLI + BC 계산 (MVP)
- [x] `src/test_chunking.py` 신규 생성
- [x] argparse CLI 인터페이스 구현 (--input, --output-dir, --llm-model)
- [x] 파싱 로직 연동 (test_parsers.py 함수 import)
- [x] Chunking 파이프라인 구현 (chunker.py 호출)
- [x] LLM API 연동 (Perplexity 계산)
- [x] BC (Boundary Clarity) 계산 및 콘솔 출력
- [x] Chunk 데이터 JSON 저장 기능
- [x] 기본 테스트 실행 및 검증 (test_1 데이터)

**Deliverable**: `python -m src.test_chunking --input PDF` 기본 실행

### Phase 2: CS 계산 + 그래프 구성
- [x] Edge weight 계산 로직 구현
- [x] Complete Graph 구성 및 threshold K 필터링
- [x] Incomplete Graph (Sequential) 구성
- [x] CS (Chunk Stickiness) 계산 - Structural Entropy
- [x] `--graph-type` 옵션 추가 (complete/incomplete)
- [x] `--threshold-k` 옵션 추가 (기본: 0.8)
- [x] 파서별 비교 테이블 콘솔 출력
- [x] evaluation.json 저장 구현

**Deliverable**: BC + CS 완전 평가 시스템

### Phase 3: Advanced Features
- [x] 기존 파싱 결과 파일 입력 (`--parsed-files`, `--parsed-dir`)
  - 포맷: `{parser}_output.{md|txt}` 패턴 인식
- [x] 다중 Chunking 전략 비교 (`--strategies`)
- [x] test_parsers.py 옵션 호환 (--skip-vlm, --skip-docling)
- [x] 결과 요약 Markdown (README.md) 생성
- [x] 에러 핸들링 및 사용자 친화적 메시지
- [x] pyproject.toml 스크립트 등록
  ```toml
  [project.scripts]
  test-chunking = "src.test_chunking:main"
  ```

**Deliverable**: 완전한 CLI 도구

### Phase 4: Documentation & Testing
- [x] README.md 업데이트 (Chunking CLI 사용법 추가)
- [x] 단위 테스트 작성 (tests/test_chunking_cli.py) - 32개 테스트 통과
- [x] 벤치마크 테스트 케이스 실행
  | ID | 데이터 | 결과 |
  |----|--------|------|
  | TC-001 | test_1 | BC = 0.9567 (8 chunks) |
  | TC-002 | test_2 | 1 chunk (평가 스킵 - 짧은 영수증) |
  | TC-003 | test_3 | BC = 0.8864 (28 chunks) |
- [x] 모듈 임포트 호환성 수정 (src.xxx / xxx 둘 다 지원)

**Deliverable**: 문서화 및 테스트 완료

## Progress

| Metric | Value |
|--------|-------|
| Total Tasks | 28/28 |
| Current Phase | All Phases Completed |
| Status | completed |

## Execution Log

| Timestamp | Phase | Task | Status |
|-----------|-------|------|--------|
| 2026-01-26 | Phase 1-3 | Core implementation | completed |
| 2026-01-26 | Phase 1 | src/test_chunking.py 생성 | completed |
| 2026-01-26 | Phase 1 | src/chunking/metrics.py 재작성 (MoC 기반) | completed |
| 2026-01-26 | Phase 2 | BC/CS 메트릭 구현 | completed |
| 2026-01-26 | Phase 3 | pyproject.toml 스크립트 등록 | completed |
| 2026-01-26 | Phase 1 | CLI 테스트 실행 및 검증 | completed |
| 2026-01-26 | Phase 4 | README.md 업데이트 | completed |
| 2026-01-26 | Phase 4 | 단위 테스트 작성 (32개 테스트 통과) | completed |
| 2026-01-26 | Phase 4 | 벤치마크 테스트 실행 (TC-001~003) | completed |
| 2026-01-26 | Phase 4 | 모듈 임포트 호환성 수정 | completed |
| 2026-01-26 | Phase 3 | --strategies 다중 전략 비교 옵션 | completed |
| 2026-01-26 | ALL | 전체 작업 완료 | **DONE** |

## File Mapping

| Task | Target File |
|------|-------------|
| CLI 생성 | `src/test_chunking.py` (신규) |
| 메트릭 구현 | `src/chunking/metrics.py` (재작성) |
| 테스트 | `tests/test_chunking_cli.py` (신규) |
| 문서 | `README.md` (수정) |
| 결과 | `results/chunks/{timestamp}/` |

## Technical Notes

### MoC 논문 핵심 공식 (arXiv:2503.09600v2)

```python
# Boundary Clarity (BC) - 높을수록 좋음
def boundary_clarity(q, d, lm):
    """
    BC(q, d) = ppl(q|d) / ppl(q)
    - q: 다음 청크
    - d: 현재 청크
    - 1에 가까움: 독립적 (좋음)
    - 0에 가까움: 상호의존적 (나쁨)
    """
    ppl_q = lm.perplexity(q)
    ppl_q_given_d = lm.perplexity(q, context=d)
    return ppl_q_given_d / ppl_q

# Edge Weight for CS Graph
def edge_weight(q, d, lm):
    """
    Edge(q, d) = (ppl(q) - ppl(q|d)) / ppl(q)
    - 1에 가까움: 높은 상관관계
    - 0에 가까움: 독립적
    """
    ppl_q = lm.perplexity(q)
    ppl_q_given_d = lm.perplexity(q, context=d)
    return (ppl_q - ppl_q_given_d) / ppl_q

# Chunk Stickiness (CS) - 낮을수록 좋음
def chunk_stickiness(graph):
    """
    CS(G) = -Σ (hi/2m) · log2(hi/2m)  (Structural Entropy)
    - hi: 노드 i의 degree
    - m: 총 edge 수
    """
    import math
    m = graph.number_of_edges()
    if m == 0:
        return 0.0
    entropy = 0.0
    for node in graph.nodes():
        h_i = graph.degree(node)
        if h_i > 0:
            p = h_i / (2 * m)
            entropy -= p * math.log2(p)
    return entropy
```

### 논문 벤치마크 참고값 (Qwen2.5-7B)

| Chunking Method | BC (↑) | CS_complete (↓) | CS_incomplete (↓) | ROUGE-L |
|-----------------|--------|-----------------|-------------------|---------|
| Fixed | 0.8049 | 2.421 | 1.898 | 0.4213 |
| Llama_index | 0.8455 | 2.250 | 1.483 | 0.4326 |
| Semantic | 0.8140 | 2.325 | 1.650 | 0.4131 |
| LLM (Qwen2.5-14B) | 0.8641 | 2.125 | 1.438 | 0.4351 |

**Pearson 상관계수** (BC/CS vs RAG 성능):
- BC ↔ ROUGE-L: **0.8776** (강한 양의 상관)
- CS_c ↔ ROUGE-L: **-0.7453** (강한 음의 상관)
- CS_i ↔ ROUGE-L: **-0.6663** (중간 음의 상관)

### 재사용할 기존 코드

```python
# src/test_parsers.py
- test_vlm_parser()
- test_ocr_text_parser()
- test_ocr_image_parser()
- normalize_text()
- detect_file_format()
- FileFormat enum

# src/chunking/chunker.py
- ChunkerConfig
- create_chunker()
- RecursiveCharacterChunker
- SemanticChunker
- HierarchicalChunker
```

### CLI 인자 정의

```
--input, -i         : 입력 파일 (PDF, 이미지, HWP) - GT 불필요
--output-dir, -o    : 출력 디렉토리 (기본: results/chunks/)
--strategy          : Chunking 전략 (recursive_character)
--chunk-size        : Chunk 크기 (기본: 500)
--chunk-overlap     : Overlap (기본: 50)
--strategies        : 다중 전략 비교 (콤마 구분)
--parsed-files      : 기존 파싱 결과 파일들
--parsed-dir        : 기존 파싱 결과 디렉토리
--skip-vlm          : VLM 파서 스킵
--skip-docling      : Docling 파서 스킵
--llm-model         : Perplexity 계산용 LLM (기본: qwen2.5-7b)
--graph-type        : CS 그래프 유형 (complete/incomplete, 기본: incomplete)
--threshold-k       : Edge 필터링 threshold (기본: 0.8)
--verbose, -v       : 상세 출력
```

### 핵심 차이점: 기존 vs MoC 방식

| 항목 | 기존 metrics.py | MoC (새로 구현) |
|------|----------------|----------------|
| **BC/BS** | 위치 매칭 (F1/Recall) | Perplexity 기반 |
| **CS** | 임베딩 유사도 | Structural Entropy |
| **GT 필요** | Yes | **No** |
| **의존성** | sentence-transformers | LLM API |
| **프로덕션 적합** | ❌ | ✅ |
