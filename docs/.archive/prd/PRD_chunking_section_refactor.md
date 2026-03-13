# Chunking Test Section Refactoring PRD

> **Version**: 1.1
> **Created**: 2026-01-27
> **Updated**: 2026-01-27
> **Status**: Draft (Digging 완료)
> **Reference**: [02_log26_Peter.pdf](../reference/02_log26_Peter.pdf), [PRD_streamlit_dashboard.md](./PRD_streamlit_dashboard.md)

## 1. Overview

### 1.1 Problem Statement

현재 Chunking Test Section에 두 가지 핵심 문제가 있음:

**문제 1: 파서별 Chunking 품질 미분리**
- 현재: 전략(Fixed/Sentence/Semantic)별로만 평가
- 문제: **동일 문서를 다른 파서로 파싱한 결과**에 동일 Chunking 전략을 적용하면 BC/CS가 다름
- 필요: 파서별 × 전략별 Chunking 품질 비교

**문제 2: BC/CS 시각화 방식이 MoC 논문과 다름**

| 항목 | 현재 구현 | Peter 발표 (MoC 기반) |
|------|---------|---------------------|
| BC 시각화 | Scatter Plot (0~1 고정) | Document Flow 시계열 차트 |
| CS 시각화 | Scatter Plot (0~1 고정) | Avg. Intra-chunk Similarity (Mean±Std) |
| CS 해석 | 낮을수록 좋음 (맞음) | 낮을수록 좋음 (동일) |
| 전략 | Fixed, Sentence, Semantic | + **Structuring** 전략 추가 |
| 파서별 비교 | ❌ 없음 | ❌ 없음 (우리만의 확장) |

### 1.2 BC/CS 개념 비교 분석

#### 현재 구현의 BC/CS 정의
```
BC (Boundary Clarity): 0~1, 높을수록 좋음
- 청크 간 의미적 독립성
- 경계가 의미적으로 타당한지 평가

CS (Chunk Stickiness): 0~1, 낮을수록 좋음
- 청크 내부 응집도
- 이상적으로 0에 가까워야 함
```

#### MoC 논문 (Peter 발표) 기반 정의
```
BC (Boundary Clarity):
- "문서에서 주제가 바뀌는 지점을 Chunking이 잘 끊어주고 있는지"
- Sentence Index (Document Flow)에 따른 시계열로 표현
- 특정 위치에서 경계 품질을 직접 확인 가능

CS (Chunk Stickiness) = Avg. Intra-chunk Similarity:
- "하나의 Chunk 안에 문장들이 같은 이야기를 하고 있는지"
- 청크 내 문장 간 유사도의 평균
- Mean ± Std로 표현하여 분산 정보 제공
- Structuring 전략은 N/A (구조 기반이므로 의미 다름)
```

### 1.3 왜 다른가?

| 차이점 | 원인 | 영향 |
|--------|------|------|
| **BC 시각화** | 우리는 청크별 단일 값, MoC는 문서 흐름 기반 | 어느 위치에서 경계 문제가 발생하는지 알 수 없음 |
| **CS 표현** | 우리는 0~1 정규화, MoC는 실제 유사도 값 | 실제 유사도 분포와 표준편차 정보 손실 |
| **Quadrant 분석** | 우리 고유 해석 | MoC에는 없지만 직관적이어서 유지 가치 있음 |
| **Structuring 전략** | 미구현 | Document 구조 기반 청킹 평가 불가 |

### 1.4 Goals
- **파서별 Chunking 품질 비교** (핵심 확장)
  - 동일 문서에 대해 각 파서의 파싱 결과로 Chunking 평가
  - "파싱 품질이 Chunking 품질에 미치는 영향" 측정
- MoC 논문 기반 BC/CS 시각화 방식 도입
- Document Flow 기반 BC 시계열 차트 추가
- CS를 Avg. Intra-chunk Similarity (Mean±Std)로 표현
- Structuring 전략 지원
- 기존 Quadrant 분석 유지 (직관적 해석에 유용)
- CLI (`test_parsers.py`) 연동으로 실제 BC/CS 데이터 생성

### 1.5 Non-Goals (Out of Scope)
- 실시간 청킹 실행
- 청킹 파라미터 조정 UI
- 임베딩 모델 변경

### 1.6 Scope

| 포함 | 제외 |
|------|------|
| BC Document Flow 차트 | 실시간 청킹 |
| CS Mean±Std 막대 그래프 | 임베딩 모델 선택 |
| Structuring 전략 지원 | 파라미터 튜닝 UI |
| 기존 Quadrant 유지 | Lazy Chunking 구현 |

## 2. User Stories

### 2.1 Primary User
연구원으로서, **각 파서의 파싱 결과**에 대해 청킹 전략별 BC/CS를 MoC 논문 방식으로 시각화하여:
1. 파싱 품질이 Chunking 품질에 미치는 영향을 확인하고
2. 어느 위치에서 경계 문제가 발생하는지 진단하고 싶다.

### 2.2 Acceptance Criteria (Gherkin)

```gherkin
Scenario: 파서별 Chunking 품질 비교 (핵심)
  Given 각 파서(VLM, pdfplumber, Docling)의 파싱 결과가 있음
  When Chunking Test 탭을 선택함
  Then 파서 선택 드롭다운이 표시됨
  And 선택한 파서의 파싱 결과에 대한 Chunking 평가가 표시됨
  And 파서 간 BC/CS 비교 차트가 표시됨

Scenario: Document Flow 기반 BC 시각화
  Given 청킹 결과 JSON이 로드됨
  When Chunking Test 탭을 선택함
  Then BC가 Sentence Index (Document Flow) 축을 기준으로 시계열 차트로 표시됨
  And 전략별로 색상이 구분되어 비교 가능함

Scenario: CS Mean±Std 표현
  Given 청킹 결과 JSON이 로드됨
  When Chunking Test 탭을 선택함
  Then CS가 "Avg. Intra-chunk Similarity" 제목으로 표시됨
  And 각 전략별 평균값과 표준편차가 막대 그래프로 표시됨
  And Structuring 전략은 "N/A" 또는 별도 표기됨

Scenario: Structuring 전략 지원
  Given JSON에 "Structuring" 전략 결과가 포함됨
  When 대시보드를 로드함
  Then Structuring 전략이 다른 전략들과 함께 표시됨
  And BC는 표시되고 CS는 "N/A"로 표시됨
```

## 3. Functional Requirements

| ID | Requirement | Priority | Dependencies |
|----|------------|----------|--------------|
| FR-001 | **파서별 Chunking 결과 구조** (JSON 스키마 확장) | P0 (Must) | - |
| FR-002 | **파서별 BC/CS 비교 차트** | P0 (Must) | FR-001 |
| FR-003 | BC Document Flow 시계열 차트 추가 | P0 (Must) | FR-001 |
| FR-004 | CS를 Avg. Intra-chunk Similarity로 표현 | P0 (Must) | - |
| FR-005 | CS Mean±Std 막대 그래프 | P0 (Must) | FR-004 |
| FR-006 | CLI에서 BC/CS 데이터 생성 (`test_parsers.py` 연동) | P0 (Must) | FR-001 |
| FR-007 | Structuring 전략 지원 | P1 (Should) | FR-003, FR-005 |
| FR-008 | 기존 BC-CS Quadrant Scatter 유지 | P1 (Should) | - |
| FR-009 | 전략별/파서별 색상 통일 | P1 (Should) | - |
| FR-010 | 차트 PNG 다운로드 | P2 (Could) | FR-003, FR-005 |

## 4. Non-Functional Requirements

### 4.1 Performance
- 차트 렌더링: < 1초
- 100개 청크 시 Document Flow 차트 성능 유지

### 4.2 Compatibility
- 기존 JSON 스키마와 호환 (새 필드 optional)
- 기존 Quadrant 분석 유지

## 5. Technical Design

### 5.1 JSON 스키마 확장 (파서별 Chunking 결과)

**핵심 변경**: `chunking_results`를 **파서별 × 전략별** 구조로 확장

```json
{
  "chunking_results": {
    "VLM (Qwen3-VL)": {
      "parser": "VLM (Qwen3-VL)",
      "test_id": "test_1",
      "strategies": [
        {
          "strategy": "Semantic",
          "params": {"similarity_threshold": 0.7},
          "chunks": [
            {
              "id": 1,
              "bc": 0.92,
              "cs": 0.45,
              "length": 623,
              "tokens": 156,
              "sentence_indices": [0, 1, 2, 3]
            }
          ],
          "mean_bc": 0.878,
          "mean_cs": 0.490,
          "std_cs": 0.12,
          "std_bc": 0.08,
          "bc_by_sentence": [
            {"sentence_idx": 0, "bc": 0.85},
            {"sentence_idx": 1, "bc": 0.72}
          ]
        },
        {
          "strategy": "Fixed",
          "params": {"chunk_size": 512, "overlap": 50},
          "chunks": [...],
          "mean_bc": 0.608,
          "mean_cs": 0.822,
          "std_cs": 0.15,
          "std_bc": 0.18,
          "bc_by_sentence": [...]
        }
      ]
    },
    "pdfplumber": {
      "parser": "pdfplumber",
      "test_id": "test_1",
      "strategies": [...]
    },
    "Docling (RapidOCR)": {
      "parser": "Docling (RapidOCR)",
      "test_id": "test_1",
      "strategies": [...]
    }
  }
}
```

**구조 설명**:
- 최상위: 파서명을 키로 사용
- 각 파서 안에 `strategies` 배열
- 각 전략에 `bc_by_sentence` (Document Flow), `std_cs` (표준편차) 포함

**Structuring 전략 처리**:
```json
{
  "strategy": "Structuring",
  "params": {"method": "ToC-based"},
  "chunks": [...],
  "mean_bc": 0.95,
  "mean_cs": null,    // N/A - 구조 기반이라 의미 없음
  "std_cs": null,
  "bc_by_sentence": [...]  // BC는 유효
}
```

### 5.2 차트 컴포넌트 설계

#### 5.2.1 파서별 BC/CS 비교 차트 (신규 - 핵심)
```python
def create_parser_chunking_comparison(
    chunking_results: Dict[str, Dict],
    strategy: str = "Semantic",
    title: str = "Parser별 Chunking 품질 비교",
    height: int = 400,
) -> go.Figure:
    """
    X축: Parser
    Y축: BC (막대) / CS (막대, 다른 색)
    파서별로 동일 전략의 BC/CS 비교
    """
    parsers = list(chunking_results.keys())
    bc_values = []
    cs_values = []
    bc_stds = []
    cs_stds = []

    for parser, data in chunking_results.items():
        strategy_data = next(
            (s for s in data.get("strategies", []) if s["strategy"] == strategy),
            None
        )
        if strategy_data:
            bc_values.append(strategy_data.get("mean_bc", 0))
            cs_values.append(strategy_data.get("mean_cs") or 0)
            bc_stds.append(strategy_data.get("std_bc", 0))
            cs_stds.append(strategy_data.get("std_cs") or 0)
        else:
            bc_values.append(0)
            cs_values.append(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="BC (↑)", x=parsers, y=bc_values,
        error_y=dict(type='data', array=bc_stds, visible=True),
        marker_color=COLORS["good"],
    ))
    fig.add_trace(go.Bar(
        name="CS (↓)", x=parsers, y=cs_values,
        error_y=dict(type='data', array=cs_stds, visible=True),
        marker_color=COLORS["warning"],
    ))

    fig.update_layout(barmode="group", title=title, height=height)
    return fig
```

#### 5.2.2 BC Document Flow Chart (신규)
```python
def create_bc_document_flow(
    strategies_data: List[Dict],
    title: str = "Boundary Clarity by Document Flow",
    height: int = 400,
) -> go.Figure:
    """
    X축: Sentence Index (Document Flow)
    Y축: Boundary Clarity
    전략별 라인 차트

    Args:
        strategies_data: 특정 파서의 strategies 배열
    """
    fig = go.Figure()

    for strategy_data in strategies_data:
        strategy = strategy_data.get("strategy", "Unknown")
        bc_by_sentence = strategy_data.get("bc_by_sentence", [])

        if bc_by_sentence:
            fig.add_trace(go.Scatter(
                x=[d["sentence_idx"] for d in bc_by_sentence],
                y=[d["bc"] for d in bc_by_sentence],
                mode='lines+markers',
                name=strategy,
                line=dict(color=STRATEGY_COLORS.get(strategy, "#888"), width=2),
                marker=dict(size=6),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Sentence Index (Document Flow)",
        yaxis_title="Boundary Clarity",
        height=height,
    )
    return fig
```

#### 5.2.3 CS Mean±Std Bar Chart (신규)
```python
def create_cs_mean_std_bar(
    strategies_data: List[Dict],
    title: str = "Chunk Stickiness (Mean ± Std)",
    height: int = 350,
) -> go.Figure:
    """
    X축: Strategy
    Y축: Avg. Intra-chunk Similarity
    Error bar로 표준편차 표시
    Structuring 전략은 N/A로 표시

    Args:
        strategies_data: 특정 파서의 strategies 배열
    """
    strategies = []
    means = []
    stds = []
    colors = []

    for strategy_data in strategies_data:
        strategy = strategy_data.get("strategy", "Unknown")
        mean_cs = strategy_data.get("mean_cs")
        std_cs = strategy_data.get("std_cs")

        strategies.append(strategy)

        if mean_cs is None:  # Structuring 등 N/A 케이스
            means.append(0)
            stds.append(0)
            colors.append("#CCCCCC")  # 회색으로 표시
        else:
            means.append(mean_cs)
            stds.append(std_cs or 0)
            colors.append(STRATEGY_COLORS.get(strategy, "#888"))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=strategies,
        y=means,
        error_y=dict(type='data', array=stds, visible=True),
        marker_color=colors,
        text=[f"{m:.3f}" if m > 0 else "N/A" for m in means],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Strategy",
        yaxis_title="Avg. Intra-chunk Similarity (↓ Lower is better)",
        height=height,
    )
    return fig
```

### 5.3 UI 레이아웃 (리팩토링 후)

```
┌─────────────────────────────────────────────────────────────────┐
│  📦 Chunking Quality Analysis                                   │
├─────────────────────────────────────────────────────────────────┤
│  [Metrics 정의 Expander]                                        │
├─────────────────────────────────────────────────────────────────┤
│  ### 🔄 Parser Selection (NEW)                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Parser: [VLM (Qwen3-VL) ▼] [pdfplumber] [Docling]      │   │
│  │  Test:   [Test 1 ▼]                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ### 📊 Parser별 Chunking 품질 비교 (NEW - 핵심)                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  [Grouped Bar: Parser × (BC, CS) for Semantic strategy] │   │
│  │  X: Parser (VLM, pdfplumber, Docling)                   │   │
│  │  Y: BC (green), CS (orange) with error bars             │   │
│  │  "어떤 파서의 파싱 결과가 Chunking에 유리한가?"          │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ### 📊 Overview (선택된 파서 기준)                              │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐     │
│  │ Total       │ Fixed       │ Sentence    │ Semantic    │     │
│  │ Chunks: 14  │ BC:0.61     │ BC:0.84     │ BC:0.88     │     │
│  │             │ CS:0.82±0.1 │ CS:0.67±0.08│ CS:0.49±0.12│     │
│  └─────────────┴─────────────┴─────────────┴─────────────┘     │
├─────────────────────────────────────────────────────────────────┤
│  ### 📈 BC by Document Flow (NEW)                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  [Line Chart: Sentence Index vs BC for each strategy]   │   │
│  │  X: Sentence Index (0, 1, 2, ...)                       │   │
│  │  Y: Boundary Clarity                                     │   │
│  │  Lines: Fixed (blue), Sentence (green), Semantic (orange)│   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ### 📊 Chunk Stickiness (Mean ± Std) (NEW)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  [Bar Chart with Error Bars]                            │   │
│  │  X: Strategy                                             │   │
│  │  Y: Avg. Intra-chunk Similarity                          │   │
│  │  Error bars show standard deviation                      │   │
│  │  Structuring: N/A (회색)                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ### 🎯 BC–CS Quadrant Analysis (기존 유지)                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  [Scatter Plot: BC vs CS per chunk]                     │   │
│  │  Quadrant labels: 이상적, Fragmentation, Over-merge, etc│   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ### 🔍 Quadrant 해석 가이드                                    │
│  (기존 테이블 유지)                                              │
├─────────────────────────────────────────────────────────────────┤
│  ### 📦 Strategy Details (Expanders)                           │
│  [Fixed] [Sentence] [Semantic] [Structuring]                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 파일 구조

```
src/
├── dashboard_analysis.py      # 메인 (Chunking 섹션 수정)
├── dashboard/
│   ├── __init__.py            # 신규 함수 export 추가
│   ├── data_loader.py         # chunking 스키마 확장 지원
│   ├── charts.py              # 신규 차트 함수 추가
│   │   ├── create_parser_chunking_comparison()  # NEW
│   │   ├── create_bc_document_flow()
│   │   └── create_cs_mean_std_bar()
│   └── styles.py
├── test_parsers.py            # BC/CS 계산 로직 추가 (Phase 5)
├── chunking_evaluator.py      # NEW: Chunking 평가 모듈 (Phase 5)
results/
├── parsing_results.json       # chunking_results 스키마 확장
```

### 5.5 색상 상수 확장

```python
# dashboard_analysis.py 또는 styles.py

STRATEGY_COLORS = {
    "Fixed": "#6366F1",      # 인디고
    "Sentence": "#10B981",   # 에메랄드
    "Semantic": "#F59E0B",   # 앰버
    "Structuring": "#8B5CF6", # 보라 (NEW)
}

# 파서 색상은 기존 PARSER_COLORS 사용
```

### 5.6 data_loader.py 호환성 처리

```python
def get_chunking_data(data: Dict[str, Any]) -> Dict[str, Dict]:
    """
    기존 JSON과 새 JSON 모두 지원

    기존 형식: chunking_results가 List
    새 형식: chunking_results가 Dict (파서별)
    """
    if "error" in data:
        return {}

    chunking_results = data.get("chunking_results", {})

    # 기존 형식 (List) 감지 - 하위 호환
    if isinstance(chunking_results, list):
        result = {}
        for item in chunking_results:
            strategy = item.get("strategy", "unknown")
            result[strategy] = {
                "params": item.get("params", {}),
                "chunks": item.get("chunks", []),
                "mean_bc": item.get("mean_bc", 0.0),
                "mean_cs": item.get("mean_cs", 0.0),
                "std_cs": item.get("std_cs"),  # None if not exists
                "std_bc": item.get("std_bc"),
                "bc_by_sentence": item.get("bc_by_sentence", []),
            }
        return {"_legacy": {"strategies": list(result.values())}}

    # 새 형식 (Dict, 파서별)
    return chunking_results
```

## 6. Implementation Phases

### Phase 1: JSON 스키마 확장 + 샘플 데이터
- [ ] 파서별 chunking_results 구조 정의
- [ ] `std_cs`, `std_bc` 필드 추가
- [ ] `bc_by_sentence` 필드 추가
- [ ] `parsing_results.json` 샘플 업데이트 (파서별 구조)
- [ ] `data_loader.py` 하위 호환성 유지 (기존 List 형식도 지원)
- [ ] `__init__.py` export 업데이트

**Deliverable**: 확장된 JSON 스키마와 샘플 데이터

### Phase 2: 신규 차트 구현
- [ ] `create_parser_chunking_comparison()` 구현 (파서별 비교)
- [ ] `create_bc_document_flow()` 구현
- [ ] `create_cs_mean_std_bar()` 구현
- [ ] `charts.py`에 추가
- [ ] `STRATEGY_COLORS`에 Structuring 추가

**Deliverable**: 신규 차트 함수들

### Phase 3: UI 레이아웃 변경
- [ ] Parser 선택 드롭다운 추가
- [ ] 파서별 Chunking 품질 비교 차트 추가
- [ ] Overview KPI에 Mean±Std 표시
- [ ] BC Document Flow 차트 섹션 추가
- [ ] CS Mean±Std 막대 그래프 섹션 추가
- [ ] 기존 Quadrant Scatter 유지 (선택된 파서 기준)
- [ ] Structuring 전략 표시 (CS: N/A, 회색)

**Deliverable**: 리팩토링된 Chunking Test 탭

### Phase 4: 내보내기 및 마무리
- [ ] 신규 차트 PNG 다운로드
- [ ] 전략별/파서별 색상 통일 검증
- [ ] 대시보드 통합 테스트

**Deliverable**: 대시보드 완성

### Phase 5: CLI 연동 (BC/CS 계산 로직)
- [ ] `chunking_evaluator.py` 모듈 생성
  - [ ] BC 계산 함수 (문장별 경계 품질)
  - [ ] CS 계산 함수 (청크 내 유사도 평균)
  - [ ] bc_by_sentence 생성 함수
- [ ] `test_parsers.py` 연동
  - [ ] 각 파서의 파싱 결과에 대해 Chunking 평가 실행
  - [ ] Semantic, Fixed, Sentence 전략 적용
  - [ ] 결과를 parsing_results.json에 저장
- [ ] 임베딩 모델 선택 (sentence-transformers 권장)
- [ ] 실제 테스트 실행 및 검증

**Deliverable**: 실제 BC/CS 데이터 생성 파이프라인

### Phase 5 상세: BC/CS 계산 알고리즘

```python
# chunking_evaluator.py

from sentence_transformers import SentenceTransformer
import numpy as np

class ChunkingEvaluator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def calculate_bc_by_sentence(
        self,
        sentences: List[str],
        chunk_boundaries: List[int],  # 청크 시작 문장 인덱스
    ) -> List[Dict]:
        """
        각 문장 위치에서 경계 품질 계산

        BC = 경계 전후 문장 간 유사도 차이
        경계에서 유사도가 급격히 떨어지면 BC 높음
        """
        embeddings = self.model.encode(sentences)
        bc_by_sentence = []

        for i, sentence in enumerate(sentences):
            if i == 0 or i == len(sentences) - 1:
                bc = 0.5  # 양 끝은 중립
            else:
                # 전후 문장과의 유사도 계산
                sim_prev = cosine_similarity(embeddings[i], embeddings[i-1])
                sim_next = cosine_similarity(embeddings[i], embeddings[i+1])

                # 경계 위치인 경우
                is_boundary = i in chunk_boundaries

                if is_boundary:
                    # 경계에서는 전 문장과 유사도가 낮고,
                    # 다음 문장과 유사도가 높으면 좋은 경계
                    bc = (1 - sim_prev + sim_next) / 2
                else:
                    # 비경계에서는 양쪽과 유사도가 높아야 함
                    bc = (sim_prev + sim_next) / 2

            bc_by_sentence.append({
                "sentence_idx": i,
                "bc": float(bc),
                "is_boundary": i in chunk_boundaries,
            })

        return bc_by_sentence

    def calculate_cs(self, chunk_sentences: List[str]) -> float:
        """
        청크 내 문장들의 평균 유사도 (Intra-chunk Similarity)

        CS가 낮을수록 → 문장들이 서로 다른 주제 → 좋은 청킹
        CS가 높을수록 → 문장들이 비슷 → Over-merge 가능성
        """
        if len(chunk_sentences) < 2:
            return 0.0

        embeddings = self.model.encode(chunk_sentences)
        similarities = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        return float(np.mean(similarities))
```

**Deliverable**: BC/CS 자동 계산 파이프라인

## 7. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| 파서별 Chunking 비교 차트 | 모든 파서 (3개) 표시 | 수동 검증 |
| BC Document Flow 차트 렌더링 | < 1초 | 브라우저 DevTools |
| CS Mean±Std 정보 가시성 | 모든 전략 (4개) 표시 | 수동 검증 |
| 기존 Quadrant 호환성 | 유지 | 기능 테스트 |
| CLI BC/CS 계산 정확도 | MoC 논문 방식 준수 | 알고리즘 리뷰 |

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `bc_by_sentence` 데이터 없음 | 높음 | Phase 1에서 샘플 데이터로 먼저 구현, Phase 5에서 실제 계산 |
| `std_cs` 데이터 없음 | 중간 | Error bar 없이 평균만 표시 (graceful degradation) |
| Structuring 전략 미구현 | 낮음 | CS를 N/A로 표시, 회색 처리 |
| 임베딩 모델 성능 | 중간 | all-MiniLM-L6-v2 사용 (빠르고 충분한 품질) |
| 파서별 문장 분할 차이 | 높음 | 각 파서 결과에 대해 별도 문장 분할 수행 |
| 기존 JSON 하위 호환성 | 중간 | data_loader.py에서 List/Dict 모두 지원 |

## 9. Appendix: MoC 논문 참고 자료

### 9.1 BC 계산 방식 (from PDF p.18-19)
- Document를 문장 단위로 분할
- 각 문장 위치에서 청킹 경계가 얼마나 적절한지 측정
- "문서에서 주제가 바뀌는 지점을 Chunking이 잘 끊어주고 있는지"

### 9.2 CS 계산 방식 (from PDF p.18-19)
- Chunk 내 문장들 간의 평균 유사도 (Intra-chunk Similarity)
- "하나의 Chunk 안에 문장들이 같은 이야기를 하고 있는지"
- 낮을수록 좋음 (각 청크가 단일 주제에 집중)

### 9.3 Structuring 전략 (from PDF p.18-19, 47-48)
- 문서 구조 (ToC, Heading) 기반 청킹
- BC는 높지만 CS 개념이 다르게 적용됨 (N/A)
- "Structuring 기법으로 Recall 85%" 달성 사례

## 10. 적용 가능성 결론

| 항목 | 적용 여부 | 이유 |
|------|---------|------|
| **파서별 Chunking 비교** | **적용** (핵심 확장) | 파싱 품질 → Chunking 품질 영향 측정 |
| BC Document Flow 차트 | **적용** | 경계 문제 위치 파악에 필수적 |
| CS Mean±Std 표현 | **적용** | 분산 정보가 품질 판단에 중요 |
| Structuring 전략 | **적용** | 구조 기반 청킹 평가 필요 |
| 기존 Quadrant 분석 | **유지** | 직관적 해석에 유용, MoC에는 없지만 가치 있음 |
| CLI 연동 (BC/CS 계산) | **적용** | 실제 데이터 생성에 필수 |

## 11. Digging 분석 결과 반영

### 수정된 항목 (v1.0 → v1.1)

| 이슈 ID | 내용 | 조치 |
|---------|------|------|
| C-1 | bc_by_sentence 데이터 생성 로직 미정의 | Phase 5 추가 (CLI 연동) |
| M-1 | 차트 함수 시그니처 불일치 | DataFrame 대신 List[Dict] 사용으로 통일 |
| M-2 | Structuring CS 처리 미정의 | N/A 케이스 명세 추가 (회색 막대, 0값) |
| M-3 | data_loader.py 호환성 | 기존 List + 새 Dict 모두 지원 |
| M-4 | __init__.py 업데이트 누락 | Phase 1에 task 추가 |
| M-5 | STRATEGY_COLORS 확장 | Structuring 색상 추가 (#8B5CF6) |
| NEW-1 | 파서별 Chunking 비교 | FR-001, FR-002 추가, JSON 스키마 변경 |
