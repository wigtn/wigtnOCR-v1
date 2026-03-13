# PRD Analysis Report: Chunking Evaluation CLI

> **Analyzed**: 2026-01-26
> **PRD Version**: 1.0
> **Analyst**: Claude (digging skill)

---

## Executive Summary

| Category | Issues Found | Severity Distribution |
|----------|-------------|----------------------|
| **Completeness** | 3 | 1 Critical, 2 Major |
| **Feasibility** | 2 | 1 Major, 1 Minor |
| **Consistency** | 2 | 2 Major |
| **Integration** | 4 | 1 Critical, 2 Major, 1 Minor |
| **Total** | 11 | 2 Critical, 7 Major, 2 Minor |

**Overall Assessment**: PRD는 전반적으로 잘 작성되었으나, 기존 `test_parsers.py`와의 통합 전략 및 메트릭 정의 불일치에 대한 명확한 결정이 필요합니다.

---

## 1. Critical Issues (Must Fix Before Implementation)

### CRIT-001: BS 계산 방식의 이중 정의

**Location**: PRD Section 11.2, `src/chunking/metrics.py`

**현재 상황**:
- `metrics.py` (line 75-135): Boundary **위치 매칭** 기반 BS 계산
  ```python
  # 현재 구현: GT 경계 위치와 예측 경계 위치의 F1/Recall
  BS = aligned_boundaries / total_gt_boundaries
  ```
- PRD Section 11.1: Perplexity 기반 **Boundary Clarity** 정의
  ```
  BC(q, d) = ppl(q|d) / ppl(q)
  ```

**문제점**:
1. "Boundary Score (BS)"와 "Boundary Clarity (BC)"가 완전히 다른 메트릭
2. PRD의 FR-003이 어느 방식을 의미하는지 불명확
3. 논문 비교 시 용어 혼란 발생 가능

**권장 조치**:
```markdown
명확한 용어 구분:
- BS (Boundary Score): 현재 metrics.py 방식 (위치 기반, 빠름)
- BC (Boundary Clarity): 논문 방식 (perplexity 기반, 느림)

PRD에 명시적으로 추가:
| 메트릭 | 계산 방식 | 사용 시점 |
|--------|----------|----------|
| BS | 위치 매칭 (F1/Recall) | 기본 (Phase 1-2) |
| BC | Perplexity 기반 | `--academic` 옵션 (Phase 5) |
```

---

### CRIT-002: test_parsers.py 통합 전략 미결정

**Location**: PRD Section 10 (Open Questions #3), `src/test_parsers.py`

**현재 상황**:
- PRD Open Question: "별도 파일 vs 옵션 추가?"
- `test_parsers.py`는 902줄의 완전한 CLI
- 많은 유틸리티 함수 재사용 필요

**분석된 재사용 가능 함수** (`test_parsers.py`):

| 함수 | 라인 | 재사용 필요성 |
|------|------|--------------|
| `detect_file_format()` | 53-66 | **필수** |
| `normalize_text()` | 138-179 | **필수** |
| `get_tokenizer()` | 186-222 | 선택적 |
| `test_vlm_parser()` | 320-410 | **필수** |
| `test_ocr_text_parser()` | 413-451 | **필수** |
| `test_ocr_image_parser()` | 454-493 | **필수** |
| `FileFormat` enum | 44-51 | **필수** |
| `save_results_to_files()` | 563-661 | 참조 (패턴 재사용) |

**권장 조치**:
```
Option A: 별도 파일 (권장)
├── src/test_parsers.py (기존 유지)
├── src/test_chunking.py (신규)
└── src/_parser_utils.py (공통 유틸리티 추출)

리팩토링 작업:
1. 공통 함수를 _parser_utils.py로 추출
2. test_parsers.py에서 import
3. test_chunking.py에서 import
```

---

## 2. Major Issues (Should Fix)

### MAJ-001: CS 계산 방식 불일치

**Location**: `metrics.py` line 172-218, PRD Section 11.1

**현재 구현**:
```python
# metrics.py: 임베딩 유사도 기반
def _calculate_single_chunk_coherence(text, model):
    sentences = split(text)
    embeddings = model.encode(sentences)
    return avg(pairwise_cosine_similarity(embeddings))
```

**논문 정의**:
```python
# Structural Entropy 기반
CS(G) = -Σ (hi/2m) · log2(hi/2m)
```

**문제점**:
- 두 방식이 측정하는 것이 다름
- 현재: 청크 **내부** 문장 간 coherence (높을수록 좋음)
- 논문: 청크 **간** 의존성 그래프 엔트로피 (낮을수록 좋음)

**권장 조치**:
- 현재 메트릭을 "Chunk Coherence (CC)"로 명명
- 논문 메트릭을 "Chunk Stickiness (CS)"로 구분
- 두 메트릭 모두 구현하되 명확히 구분

---

### MAJ-002: 파싱 결과 재사용 포맷 미정의

**Location**: PRD FR-006, Scenario 2

**문제점**:
- `--parsed-dir` 입력 시 어떤 포맷의 파일을 읽을지 미정의
- 현재 `test_parsers.py`의 출력 포맷과 일치해야 함

**현재 test_parsers.py 출력 분석** (line 563-661):
```
{output_dir}/
├── vlm_output.md       # VLM 결과 (markdown)
├── ocr-text_output.txt # OCR-Text 결과 (plain text)
├── ocr-image_output.md # OCR-Image 결과 (markdown)
├── evaluation.json     # 메트릭
└── README.md           # 요약
```

**권장 조치**:
PRD에 명시적 포맷 정의 추가:
```markdown
### 파싱 결과 입력 포맷

`--parsed-dir` 사용 시 다음 파일 구조 기대:
- `vlm_output.md` 또는 `vlm_output.txt`
- `ocr-text_output.md` 또는 `ocr-text_output.txt`
- `ocr-image_output.md` 또는 `ocr-image_output.txt`

파일명 패턴: `{parser-name}_output.{md|txt}`
```

---

### MAJ-003: Chunking 전략 파라미터 통합 부재

**Location**: PRD Section 5.2, `chunker.py`

**분석**:
- `chunker.py`의 `SemanticChunker`는 `embedding_model` 파라미터 필요 (line 268)
- PRD CLI에 임베딩 모델 지정 옵션 없음

**chunker.py 분석**:
```python
class SemanticChunker(TextChunker):
    def __init__(self, config, embedding_model="jhgan/ko-sroberta-multitask"):
        # 임베딩 모델은 하드코딩됨
```

**권장 조치**:
CLI 옵션 추가:
```bash
--embedding-model    # 임베딩 모델 지정 (기본: jhgan/ko-sroberta-multitask)
--similarity-threshold  # SemanticChunker의 breakpoint threshold (기본: 0.3)
```

---

### MAJ-004: Ground Truth 없는 실행 시나리오 미정의

**Location**: PRD Acceptance Criteria

**문제점**:
- 모든 시나리오가 `--gt` 옵션 필수로 가정
- GT 없이 Chunking만 수행하려는 경우 미정의

**권장 조치**:
```gherkin
Scenario: Ground Truth 없이 Chunking만 수행
  Given PDF 파일만 존재
  When `python -m src.test_chunking --input data/test.pdf` 실행
  Then 각 파서별 Chunking 수행
  And Chunk 데이터 JSON 저장
  And BS/CS 계산 스킵 (경고 메시지 출력)
```

---

### MAJ-005: 오류 처리 시나리오 부재

**Location**: PRD 전체

**누락된 에러 케이스**:
1. VLM 파서 실패 시 (GPU 없음)
2. 임베딩 모델 로드 실패 시
3. Docling 미설치 시
4. 잘못된 파일 포맷 시

**권장 조치**:
```markdown
## Error Handling

| Error | Message | Exit Code |
|-------|---------|-----------|
| VLM 실패 | "VLM 파서 실패. --skip-vlm 옵션으로 스킵 가능" | 1 |
| 임베딩 실패 | "임베딩 모델 로드 실패. --skip-cs 옵션으로 스킵 가능" | 1 |
| 모든 파서 실패 | "모든 파서가 실패했습니다" | 2 |
| 파일 없음 | "입력 파일을 찾을 수 없습니다: {path}" | 3 |
```

---

### MAJ-006: 벤치마크 데이터 미정의

**Location**: PRD Section 7 (Success Metrics)

**문제점**:
- "모든 테스트 케이스 통과" - 어떤 테스트 케이스?
- 벤치마크 대상 데이터셋 미정의

**현재 사용 가능한 테스트 데이터**:
```
data/
├── test_1/  # 한국어 정부 문서
├── test_2/  # 영수증 이미지
├── test_3/  # 영어 학술 논문 (스캔)
└── test_4_hwp/  # HWP 문서 (신규)
```

**권장 조치**:
```markdown
### 벤치마크 테스트 케이스

| ID | 데이터 | 예상 결과 |
|----|--------|----------|
| TC-001 | test_1 (한국어 PDF) | VLM BS > OCR BS |
| TC-002 | test_2 (영수증 이미지) | VLM만 파싱 성공 |
| TC-003 | test_3 (스캔 PDF) | VLM BS >> OCR BS |
| TC-004 | test_4_hwp | VLM 파싱 성공 |
```

---

### MAJ-007: pyproject.toml 스크립트 등록 상세 미정의

**Location**: PRD Phase 3, PLAN Phase 3

**현재 pyproject.toml 분석** (line 49-53):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
# 스크립트 등록 없음
```

**권장 조치**:
PRD에 추가:
```toml
[project.scripts]
test-chunking = "src.test_chunking:main"
test-parsers = "src.test_parsers:main"
```

---

## 3. Minor Issues (Nice to Fix)

### MIN-001: 타임스탬프 포맷 불일치

**Location**: PRD Section 5.3 vs test_parsers.py

- PRD: `2026-01-26T10:30:00Z` (ISO 8601)
- test_parsers.py (line 572): `%Y%m%d_%H%M%S` (간소화)

**권장 조치**: 하나로 통일 (ISO 8601 권장)

---

### MIN-002: 출력 디렉토리 기본값 불일치

**Location**: PRD vs PLAN

- PRD Section 5.5: `results/chunks/{timestamp}/`
- PLAN: `results/chunks/`

**권장 조치**: PRD 방식으로 통일 (타임스탬프 포함)

---

## 4. Integration Analysis: test_parsers.py

### 4.1 재사용 가능한 코드 패턴

```python
# test_chunking.py에서 그대로 사용 가능한 패턴

# 1. CLI 구조 (test_parsers.py:698-767)
parser = argparse.ArgumentParser(
    description="...",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="..."
)

# 2. 파서 실행 루프 (test_parsers.py:825-884)
if not args.skip_vlm:
    results["VLM"] = test_vlm_parser(...)

# 3. 결과 저장 패턴 (test_parsers.py:563-661)
# JSON + README.md 저장

# 4. 요약 출력 패턴 (test_parsers.py:664-691)
print("\n| Parser | 성공 | 시간 | 추출 길이 |")
```

### 4.2 수정 필요한 부분

```python
# test_parsers.py의 evaluate_results()는 CER/WER 전용
# test_chunking.py는 새로운 evaluate_chunking_results() 필요

def evaluate_chunking_results(
    chunks: dict[str, list[Chunk]],  # 파서별 청크
    ground_truth: str,
    skip_cs: bool = False
) -> dict:
    """청킹 결과 평가 (BS, CS)"""
    ...
```

### 4.3 권장 리팩토링 계획

```
Step 1: 공통 유틸리티 추출
─────────────────────────
src/_parser_utils.py (신규):
├── FileFormat enum
├── detect_file_format()
├── normalize_text()
├── get_tokenizer()
└── save_results_to_files() → save_json(), save_markdown()

Step 2: test_parsers.py 수정
─────────────────────────────
- from ._parser_utils import ...
- 기존 로직 유지

Step 3: test_chunking.py 생성
─────────────────────────────
- from ._parser_utils import ...
- from .test_parsers import test_vlm_parser, test_ocr_text_parser, ...
- 신규 chunking 로직
```

---

## 5. Risk Matrix Update

| Risk | Probability | Impact | Mitigation | Status |
|------|-------------|--------|------------|--------|
| 메트릭 혼란 (BS vs BC) | High | High | 명확한 용어 구분 | **PRD 수정 필요** |
| 통합 복잡도 | Medium | Medium | 단계별 리팩토링 | PLAN 업데이트 |
| 테스트 누락 | Medium | High | 테스트 케이스 정의 | **PRD 수정 필요** |
| 임베딩 의존성 | Low | Medium | --skip-cs 옵션 | 해결됨 |

---

## 6. Recommended Actions

### 즉시 수정 (Before Implementation)

1. **CRIT-001**: BS/BC 용어 명확화
   - PRD Section 3, 11에 명시적 구분 추가

2. **CRIT-002**: 통합 전략 결정
   - 별도 파일 + 공통 유틸리티 추출 방식 채택
   - PLAN Phase 1에 리팩토링 태스크 추가

### 구현 중 수정

3. **MAJ-002**: 파싱 결과 포맷 정의
4. **MAJ-004**: GT 없는 실행 시나리오 추가
5. **MAJ-005**: 에러 처리 시나리오 추가

### 구현 후 수정

6. **MAJ-006**: 벤치마크 테스트 케이스 작성
7. **MAJ-007**: pyproject.toml 스크립트 등록

---

## 7. Updated Task Plan Recommendation

```markdown
### Phase 0: 준비 (신규)
- [ ] 공통 유틸리티 모듈 추출 (`src/_parser_utils.py`)
- [ ] test_parsers.py 리팩토링 (import 변경)
- [ ] 메트릭 용어 명확화 (BS vs BC, CC vs CS)

### Phase 1: Core CLI (기존 유지)
- [ ] `src/test_chunking.py` 신규 생성
- [ ] 기본 CLI 인터페이스
- [ ] BS 계산 및 출력
- [ ] Chunk JSON 저장

### Phase 2: Complete Metrics (기존 유지)
- [ ] CC (Chunk Coherence) 계산 통합
- [ ] `--skip-cs` 옵션
- [ ] evaluation.json 저장

### Phase 3-4: (기존 유지)

### Phase 5: Academic Metrics (기존 유지)
- [ ] BC (Boundary Clarity) - Perplexity 기반
- [ ] CS (Chunk Stickiness) - Structural Entropy 기반
- [ ] `--academic` 옵션
```

---

## Appendix: Code References

| File | Lines | Description |
|------|-------|-------------|
| `test_parsers.py` | 1-902 | 기존 파싱 CLI |
| `chunker.py` | 1-428 | Chunking 전략 구현 |
| `metrics.py` | 1-345 | BS/CS 메트릭 구현 |
| `PRD_chunking_evaluation_cli.md` | 1-497 | 분석 대상 PRD |

---

**Analysis Complete**: 2026-01-26
