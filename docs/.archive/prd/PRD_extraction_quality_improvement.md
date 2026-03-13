# Fair CER Evaluation Pipeline PRD

> **Version**: 2.0 (전면 재작성)
> **Created**: 2026-01-30
> **Status**: Draft
> **Context**: 국제 학회 제출 논문의 실험 근거 — 평가 방법론의 논리적 엄밀성 필수

## 1. Overview

### 1.1 Problem Statement

현재 CER 평가가 **불공정한 비교**를 수행하고 있다.

GT(pandoc LaTeX→Markdown)와 추출 텍스트(PyMuPDF)의 **콘텐츠 구성이 구조적으로 다르다**:

| 구성 요소 | GT (pandoc) | 추출 (PyMuPDF) | 불일치 |
|----------|-------------|---------------|--------|
| 본문 텍스트 | O | O | 읽기 순서 차이 |
| 참고문헌 리스트 | **X** (없음) | **O** (전체의 5~43%) | GT에 없음 |
| 인라인 인용 | `[@key]` → normalize에서 제거 | `[1]`, `(Author, 2020)` | 형식 다름 |
| 수식 | `$E=mc^2$` → normalize에서 제거 | 깨진 유니코드/누락 | 양쪽 다름 |
| 페이지 번호 | X | O (29~152줄/문서) | 추출에만 있음 |
| 러닝 헤더 | X | O (3~14회/문서) | 추출에만 있음 |
| 저자/소속/이메일 | 부분적 (pandoc 변환) | O (전체) | 범위 다름 |
| 각주 정의 | `[^N]: text` (본문 하단) | 본문에 인라인 삽입 | 위치 다름 |
| `[Page N]` 마커 | X | O (15~26개/문서) | 추출에만 있음 |
| pandoc div 블록 | `:::` 태그 | X | GT에만 있음 |

**핵심 문제**: `normalize_text()`는 GT와 추출 양쪽에 동일 적용된다. 따라서 **한쪽에만 존재하는 콘텐츠를 제거하면 해당 쪽만 짧아져서 CER이 오히려 악화**된다.

실측 검증:
- 참고문헌 제거: CER 40% → **56%** (16pp 악화)
- 이메일 줄 제거: CER 40% → **48%** (8pp 악화)
- 페이지 번호 제거: CER 40% → 39.6% (0.4pp 개선, 무의미)

### 1.2 근본 원인

```
LaTeX Source (.tex)
    │
    ├──→ pandoc ──→ Markdown GT (참고문헌 없음, 수식 $...$, 인용 [@key])
    │
    └──→ pdflatex ──→ PDF ──→ PyMuPDF ──→ Extracted Text (참고문헌 있음, 수식 깨짐, 인용 [1])
```

**동일한 LaTeX 소스에서 두 경로를 거쳐 나온 텍스트가 다른 구성**을 갖는다.
공정한 비교를 위해서는 **양쪽의 평가 범위(scope)를 동일하게 정의**해야 한다.

### 1.3 Goals

1. **공정한 비교 범위 정의**: GT와 추출 텍스트에서 동일 범위의 콘텐츠만 비교
2. **재현 가능한 파이프라인**: 정규화 규칙을 명시적으로 문서화
3. **학회 논문 수준의 엄밀성**: 왜 이 정규화가 공정한지 논리적 근거 제시
4. **CER 상세 메트릭 저장**: S/D/I 분해로 오류 원인 분석 가능

### 1.4 Non-Goals

- PyMuPDF 추출 엔진 교체 (별도 연구)
- GT 재빌드 (본 PRD는 평가 파이프라인 개선에 집중)
- 새로운 메트릭 (BLEU, ROUGE 등) 추가 (별도 PRD)

## 2. Technical Design

### 2.1 설계 원칙: Scope Alignment

CER 비교의 공정성을 위해 다음 원칙을 따른다:

> **원칙 1**: 양쪽 텍스트에 **동일하게 존재하는 콘텐츠**만 비교한다.
>
> **원칙 2**: 한쪽에만 존재하는 콘텐츠는 **양쪽 모두에서** 해당 영역을 제거하되,
> 제거 불가능한 경우(한쪽에 해당 영역이 없는 경우) **제거하지 않는다**.
>
> **원칙 3**: 포맷팅 차이(마크다운 문법, 공백 등)는 양쪽에 동일 규칙으로 제거한다. (기존 구현)

이 원칙에 따른 분류:

| 구성 요소 | GT | 추출 | 양쪽 존재 | 처리 |
|----------|----|----|----------|------|
| 본문 텍스트 | O | O | O | 비교 대상 (유지) |
| 마크다운 문법 | O | X | — | 제거 (기존 구현) |
| pandoc 아티팩트 | O | X | — | 제거 (기존 구현) |
| 수식 | O | 깨짐/누락 | 부분적 | 양쪽 제거 (기존 구현, 공정) |
| 인라인 인용 `[@key]` / `[1]` | O | O | 형식 다름 | 양쪽에서 인용 패턴 제거 |
| 참고문헌 리스트 | **X** | O | **X** | **제거 불가** (원칙 2 위반) |
| 각주 정의 `[^N]:` | O | X | — | GT에서 제거 (기존 구현) |
| 페이지 번호 | X | O | X | **제거 불가** |
| 러닝 헤더 | X | O | X | **제거 불가** |
| 저자/이메일 | 부분적 | O | 부분적 | **제거 불가** (범위 불확실) |
| `[Page N]` 마커 | X | O | X | **제거 불가** |

### 2.2 결론: normalize_text()에서 추가할 수 있는 것

원칙 1~3을 엄격히 적용하면, **추가로 제거할 수 있는 것이 거의 없다**.

기존 normalize_text()에서 이미 처리한 것:
- 마크다운 문법 (헤더, bold, italic, 링크, 리스트, 테이블 파이프)
- pandoc 아티팩트 (div 블록, 참조 속성, 인용 `[@key]`, 각주 `[^N]`, 이미지 문법)
- 수식 (`$$...$$`, `$...$`)
- `[Page N]` 마커, `width=` 속성

**새로 추가 가능한 것 (양쪽에 동일하게 존재)**:

| 항목 | 근거 | 예상 효과 |
|------|------|----------|
| 숫자 기반 인용 `[1]`, `[1,2]`, `[1-3]` | 추출에만 `[1]` 형태, GT에서는 `[@key]`가 이미 제거됨. 추출의 인용 참조도 제거해야 공정 | 중간 (인용 빈도에 비례) |
| 괄호 인용 `(Author, 2020)` | 일부 논문에서 사용. GT에서는 이미 제거됨 | 낮음 |
| `---` 수평선 (페이지 구분자) | ocr_parser가 `[Page N]` 사이에 삽입. GT에도 없음 | 미미 |

### 2.3 올바른 CER 개선 경로

normalize_text() 확장만으로는 CER을 대폭 낮출 수 없다. **구조적으로 올바른 개선 경로**:

#### 경로 A: GT 재빌드 (가장 효과적, 별도 PRD)
- pandoc 변환 시 참고문헌 리스트를 GT에 포함
- `merge_bbl()` → pandoc이 `\bibliography`를 변환하도록 개선
- GT와 추출의 콘텐츠 구성을 일치시키면 CER이 자연 하락

#### 경로 B: 섹션별 CER (본 PRD에서 구현)
- "본문 CER"과 "전체 CER"을 분리
- 논문에서 "본문 텍스트 추출 정확도"를 보고할 때 본문 CER 사용
- 참고문헌 포함 전체 CER은 별도 보고

#### 경로 C: 추출 품질 실질 개선 (별도 PRD)
- PyMuPDF `get_text("blocks")` + column-aware 정렬
- Substitution 17% 감소가 핵심

## 3. Functional Requirements

| ID | Requirement | Priority | 근거 |
|----|------------|----------|------|
| FR-001 | 추출 텍스트의 숫자 기반 인용 참조 `[1]`, `[1,2]`, `[1-3]` 제거 | P0 | GT의 `[@key]`는 이미 제거됨. 추출의 `[1]`도 제거해야 양쪽 공정 |
| FR-002 | `---` 수평선 제거 | P0 | ocr_parser가 삽입한 페이지 구분자. GT에도 없음 |
| FR-003 | 섹션별 CER 분리 측정 (Body CER) | P0 | 참고문헌 미포함 본문만의 정확도를 별도 보고 |
| FR-004 | evaluation.json에 CER 상세(S/D/I/hits) 저장 | P0 | 오류 원인 분석에 필수 |
| FR-005 | 괄호 인용 `(Author et al., 2020)` 제거 | P1 | 일부 논문에서 사용 |
| FR-006 | 각주 정의 본문 제거 (`[^N]: text` 줄 전체) | P1 | GT에만 있는 콘텐츠이므로 normalize에서 제거 (기존 `[^N]` 참조만 제거, 정의 본문은 미제거) |

## 4. Detailed Design

### 4.1 FR-001: 숫자 기반 인용 참조 제거

GT에서 `[@hochreiter1997]`가 normalize에서 이미 제거되므로, 추출의 대응 패턴 `[1]`도 제거해야 공정하다.

```python
# 숫자 인용 참조: [1], [1,2], [1, 2, 3], [1-3], [12, 15-17]
result = re.sub(r'\[\d+(?:\s*[,;-]\s*\d+)*\]', '', result)
```

**주의**: `[Page 1]` 마커는 이미 앞 단계에서 제거됨. 본문의 `[Table 1]` 등은 숫자만으로 구성되지 않으므로 매칭 안 됨.

### 4.2 FR-002: 수평선 제거

```python
# --- 또는 *** 수평선
result = re.sub(r'^\s*[-*_]{3,}\s*$', '', result, flags=re.MULTILINE)
```

### 4.3 FR-003: 섹션별 CER (Body CER)

추출 텍스트에서 "References" 이후를 분리하고, GT 전체와 비교하여 **Body CER**을 산출한다.

설계 논리:
- GT에는 참고문헌이 없으므로, GT 전체 = GT 본문
- 추출에서 References 이전 = 추출 본문
- Body CER = CER(추출 본문, GT 본문)

```python
def split_body_references(text: str) -> tuple[str, str]:
    """텍스트를 본문과 참고문헌으로 분리.

    Returns:
        (body, references) — references가 없으면 ("", "")
    """
    # References/Bibliography 섹션 시작점 찾기
    match = re.search(
        r'^(?:#{0,3}\s*)?(?:References|Bibliography|REFERENCES)\s*$',
        text, re.MULTILINE
    )
    if match:
        return text[:match.start()].rstrip(), text[match.start():]
    return text, ""


def calculate_body_cer(hypothesis: str, reference: str) -> dict:
    """Body CER: 추출 본문 vs GT 본문 비교.

    추출에서 References 이후를 제외하고 본문만 비교.
    GT에는 References가 없으므로 GT 전체 = GT 본문.
    """
    # 추출에서 본문만 추출
    hyp_body, hyp_refs = split_body_references(hypothesis)

    # GT는 전체가 본문 (References 섹션 없음)
    # 단, GT에도 References가 있을 수 있으므로 동일 처리
    ref_body, ref_refs = split_body_references(reference)
    if not ref_body:
        ref_body = reference  # GT에 References 헤더가 없으면 전체 사용

    hyp_normalized = normalize_text(hyp_body)
    ref_normalized = normalize_text(ref_body)

    return calculate_cer(hyp_normalized, ref_normalized)
```

**학회 논문에서의 보고 방식**:
- Table에 **Body CER**과 **Full CER** 두 열을 모두 보고
- 본문에서 Body CER을 주 지표로 논의
- "Full CER includes References section which is absent from GT due to pandoc conversion limitations" 각주 기재

### 4.4 FR-004: CER 상세 저장

현재 `evaluate_results()` → `save_results_to_files()`에서 `cer_detail`이 계산되지만 JSON에 저장되지 않는 버그.

```python
# save_results_to_files() 내부, 기존:
eval_json["parsers"][name]["cer"] = eval_data.get("cer")

# 수정: cer_detail도 저장
if eval_data.get("cer_detail"):
    eval_json["parsers"][name]["cer_detail"] = {
        "substitutions": eval_data["cer_detail"].get("substitutions"),
        "deletions": eval_data["cer_detail"].get("deletions"),
        "insertions": eval_data["cer_detail"].get("insertions"),
        "hits": eval_data["cer_detail"].get("hits"),
    }
```

### 4.5 FR-005: 괄호 인용 제거

```python
# (Author, 2020), (Author et al., 2020), (Author & Author, 2020)
result = re.sub(r'\([A-Z][a-z]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-z]+))?(?:,?\s*\d{4}[a-z]?)\)', '', result)
```

### 4.6 FR-006: 각주 정의 본문 제거

현재 `[^N]` 참조만 제거하고, `[^N]: footnote text` 정의 줄은 남아있음.

```python
# 각주 정의 줄 전체 제거: [^1]: Some footnote text...
result = re.sub(r'^\[\^\w+\]:.*$', '', result, flags=re.MULTILINE)
```

### 4.7 처리 순서 (최종)

```
normalize_text() 내부:
  1. [기존] Pandoc div/참조속성 제거
  2. [기존] 인용 참조 [@key] 제거
  3. [신규] 숫자 인용 참조 [1], [1,2] 제거 (FR-001)
  4. [기존] 각주 참조 [^N] 제거
  5. [신규] 각주 정의 [^N]: text 줄 제거 (FR-006)
  6. [기존] 이미지 문법 → 캡션 보존
  7. [기존] 수식 제거
  8. [기존] [Page N] 마커 제거
  9. [기존] width/height 속성 제거
 10. [신규] 수평선 --- 제거 (FR-002)
 11. [신규] 괄호 인용 (Author, 2020) 제거 (FR-005)
 12. [기존] 마크다운 문법 제거
 13. [기존] 공백 정규화

evaluate_results() 추가:
 14. [신규] Body CER 계산 (FR-003)
 15. [신규] CER 상세 저장 (FR-004)
```

## 5. Implementation Phases

### Phase 1: 공정 정규화 완성
- [ ] FR-001: 숫자 인용 참조 제거
- [ ] FR-002: 수평선 제거
- [ ] FR-006: 각주 정의 줄 제거
- [ ] FR-005: 괄호 인용 제거

### Phase 2: 섹션별 CER + 상세 저장
- [ ] FR-003: `split_body_references()` + `calculate_body_cer()` 구현
- [ ] FR-003: `evaluate_results()`에서 Body CER 계산 및 출력
- [ ] FR-003: `save_results_to_files()`에서 Body CER JSON 저장
- [ ] FR-004: `save_results_to_files()`에서 cer_detail 저장

### Phase 3: 검증
- [ ] 39개 arXiv 데이터셋 재평가 실행
- [ ] Full CER / Body CER 비교 분석
- [ ] 결과 정합성 검증 (Body CER < Full CER 확인)

## 6. Success Metrics

| Metric | 현재값 | 예상값 | 측정 방법 |
|--------|--------|--------|----------|
| Full CER 평균 | 40.0% | 35~38% | `--all --skip-advanced` |
| **Body CER 평균** | 미측정 | **20~30%** | 신규 메트릭 |
| CER 상세(S/D/I) 저장 | 미저장 | 100% 저장 | evaluation.json 확인 |

> **참고**: Full CER은 GT에 참고문헌이 없는 구조적 한계로 인해 30% 이하 달성이 어렵다.
> 학회 논문에서는 Body CER을 주 지표로 사용하고, Full CER과의 차이를 통해 참고문헌 영역의 영향을 정량화한다.

## 7. 학회 논문에서의 활용

### 7.1 실험 섹션 구성

```
Table 3: Character Error Rate by Parser
┌─────────────────┬──────────┬──────────┬──────────┐
│ Parser          │ Body CER │ Full CER │ Δ (Refs) │
├─────────────────┼──────────┼──────────┼──────────┤
│ Text-Baseline   │ XX.X%    │ XX.X%    │ +X.Xpp   │
│ Image-Baseline  │ XX.X%    │ XX.X%    │ +X.Xpp   │
│ Text-Advanced   │ XX.X%    │ XX.X%    │ +X.Xpp   │
│ Image-Advanced  │ XX.X%    │ XX.X%    │ +X.Xpp   │
└─────────────────┴──────────┴──────────┴──────────┘

Note: Body CER excludes References section from extracted text
as GT does not contain bibliography entries due to the pandoc
LaTeX→Markdown conversion pipeline. Δ quantifies the CER
increase caused by the bibliography mismatch.
```

### 7.2 논문에서 정규화 정당성 기술

> "For fair comparison, both GT and extracted text undergo identical normalization:
> removal of formatting syntax, mathematical expressions, citation markers, and
> footnote definitions. We report Body CER (excluding bibliography) as the primary
> metric, since our GT is generated via pandoc from LaTeX source which does not
> preserve the formatted bibliography section."

## 8. 리스크

| 리스크 | 확률 | 대응 |
|--------|------|------|
| 숫자 인용 `[1]` 제거가 본문의 리스트 번호 `[1]` 오탐 | 낮음 | 마크다운 리스트는 `1.` 형태이므로 `[1]` 패턴과 다름 |
| Body CER에서 추출의 References 시작점 오탐 | 낮음 | "References" 단독 줄 매칭은 정확도 높음 |
| Reviewer가 Body CER만 보고하는 것에 의문 | 중간 | Full CER 병기 + 차이 정량화로 투명성 확보 |

## 9. 수정 대상 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/eval_parsers.py` | normalize_text() 확장 (FR-001,002,005,006), Body CER (FR-003), CER 상세 (FR-004) |

## 10. 후속 PRD (본 PRD 범위 외)

| 후속 PRD | 목적 | CER 영향 |
|---------|------|----------|
| GT 재빌드 | 참고문헌 포함 GT 생성 | Full CER 대폭 개선 |
| Column-aware 추출 | 2-column 읽기 순서 개선 | Body CER의 Substitution 감소 |
| 보조 메트릭 | BLEU, ROUGE 등 순서 무관 유사도 | CER 보완 |
