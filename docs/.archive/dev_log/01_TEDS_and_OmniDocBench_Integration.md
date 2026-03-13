# 개발 로그: TEDS 메트릭 및 OmniDocBench 어댑터 도입

**작업일**: 2026-01-30
**상태**: 완료

---

## 1. 목표

기존 평가 파이프라인(CER, WER, Structure F1)에 **TEDS(Tree Edit Distance Similarity)** 메트릭을 추가하고, **OmniDocBench(CVPR 2025)** 데이터셋을 사용할 수 있는 어댑터를 구축한다.

---

## 2. TEDS 메트릭 구현 (Phase 1)

### 2.1 배경

- TEDS는 **표 구조 평가의 업계 표준 메트릭** (ECCV 2020, IBM Research)
- HTML `<table>` 트리 간 편집 거리 기반 유사도 (0~1)
- 원저자 공식 구현이 `apted` 라이브러리 사용

### 2.2 구현 내용

**변경 파일**:
- `src/eval_parsers.py` — TEDS 계산 함수 추가
- `pyproject.toml` — `apted>=1.0.3`, `mistletoe>=1.4.0` 의존성 추가
- `tests/test_parsing_cli.py` — TEDS 관련 유닛테스트 18개 추가

**핵심 함수 (eval_parsers.py)**:
1. `extract_markdown_tables(text)` — Markdown에서 파이프 테이블 추출
2. `markdown_table_to_html(md_table)` — `mistletoe`로 HTML 변환
3. `_TreeNode`, `_TEDSConfig`, `_HTMLTableTreeBuilder` — HTML → 트리 파싱
4. `compute_teds(pred_html, gt_html)` — APTED 편집 거리 → TEDS 점수
5. `calculate_teds(hypothesis, reference)` — 전체 파이프라인 (Markdown → TEDS)

**evaluation.json 스키마 확장**:
```json
{
  "parsers": {
    "Parser-Name": {
      "cer": 0.51,
      "wer": 0.57,
      "structure_f1": 0.79,
      "teds": 0.85,
      "teds_detail": {
        "table_count": 4,
        "matched_tables": 3,
        "per_table_scores": [0.92, 0.81, 0.83]
      }
    }
  }
}
```

### 2.3 시행착오

**문제 1: `import json` 중복**
- `save_results_to_files()` 내부에 로컬 `import json`이 있었는데, TEDS 구현을 위해 파일 최상단에 `import json`을 추가하면서 중복 발생
- **해결**: 로컬 `import json` 제거

**문제 2: 기존 테스트 6개 실패**
- `FileFormat.IMAGE`, `FileFormat.HWP`, `FileFormat.HWPX` 속성 미존재, `VLMParser`/`ImageOCRParser` mock 대상 미존재
- **확인 결과**: TEDS 변경과 무관한 기존 문제
- **대응**: `-k` 필터로 TEDS 테스트만 분리 실행하여 검증

### 2.4 테스트 결과

18개 TEDS 테스트 전부 통과:
- `TestExtractMarkdownTables` (4건): 단일 표, 다중 표, 표 없음, separator row
- `TestMarkdownTableToHtml` (2건): 기본 변환, `<th>` 헤더
- `TestHtmlToTree` (3건): 기본 트리, 빈 HTML, 셀 텍스트 추출
- `TestComputeTeds` (5건): 동일=1.0, 상이<1.0, 셀 1개 차이, 빈 vs 비어있지 않음=0.0, 양쪽 빈=1.0
- `TestCalculateTeds` (4건): 동일 마크다운, 참조에 표 없음, 가설에 표 누락, 상세 구조

---

## 3. OmniDocBench 어댑터 구현 (Phase 2)

### 3.1 배경

- OmniDocBench (CVPR 2025): **1,355 PDF 페이지**, 9개 문서 유형, 3개 언어 (영어, 중국어, 혼합)
- GT 형식이 **JSON** (블록별 주석) → 기존 파이프라인의 **Markdown GT**와 불일치
- 어댑터가 JSON GT ↔ Markdown 변환을 담당

### 3.2 구현 파일

- `src/adapters/__init__.py` (신규, 빈 파일)
- `src/adapters/omnidocbench.py` (신규)
- `src/eval_parsers.py` — `run_omnidocbench()` 함수 추가, `--omnidocbench`, `--limit` CLI 인수 추가

### 3.3 시행착오 — 실제 데이터와의 불일치

**HuggingFace 다운로드 이슈**:
- `huggingface-cli download` 명령어가 deprecated → `hf download`로 변경
- bash에서 `\` 줄바꿈이 별도 명령어로 파싱됨 → 단일 줄 명령어로 해결
- 최종 명령어: `hf download opendatalab/OmniDocBench --repo-type dataset --local-dir data/omnidocbench`

**문제 1: `order` 필드에 `None` 값**
```python
# 에러: TypeError: '<' not supported between instances of 'NoneType' and 'int'
sorted_blocks = sorted(layout_dets, key=lambda b: b.get("order", 0))
```
- `b.get("order", 0)`은 키가 존재하되 값이 `None`인 경우 `None`을 반환
- **해결**: `b.get("order") or 0` 으로 변경

**문제 2: `category_type` 매핑 불일치 (심각)**

실제 JSON 데이터를 분석한 결과, 코드에서 가정한 타입명과 완전히 달랐음:

| 실제 데이터 | 코드에서 가정 | 건수 |
|---|---|---|
| `text_block` | `text` | 14,202 |
| `equation_isolated` | `equation` | 1,050 |
| `page_number` | (없음) | 922 |
| `text_mask`, `table_mask` | (없음) | 507 |
| `abandon` | (없음) | 674 |
| `code_txt` | (없음) | 16 |

- **해결**: `_CATEGORY_HANDLERS` 딕셔너리를 실제 데이터의 22개 category_type 전부에 맞춰 재작성
- 레거시 이름(`text`, `equation`)도 호환성을 위해 유지

**문제 3: 수식 블록에 `text` 필드 없음**

`equation_isolated` 블록은 `text` 필드가 `None`이고 `latex` 필드에 수식이 있었음:
```json
{
  "category_type": "equation_isolated",
  "text": null,
  "latex": "$$\n\\alpha< 0, \\lambda< 0\n$$"
}
```
- **해결**: `_get_equation_text()` 헬퍼 추가 — `latex` 필드 우선 사용

**문제 4: 테이블 블록에 `text` 필드 없음**

512개 테이블 블록 전부 `text` 필드가 없고 `html`만 존재:
```json
{
  "category_type": "table",
  "html": "<table>...</table>",
  "text": null  // 또는 키 자체 없음
}
```
- 코드의 `html` → markdown 변환 경로는 정상 동작하므로 문제없음
- 일반 텍스트 블록의 fallback을 위해 `_get_block_text()` 헬퍼 추가 (text → line_with_spans 순서)

**문제 5: metadata 위치 오류**

코드에서 `item.get("doc_type")`, `item.get("language")` 사용했으나 실제 위치:
```json
{
  "page_info": {
    "page_attribute": {
      "data_source": "PPT2PDF",       // ← doc_type에 해당
      "language": "simplified_chinese",
      "layout": "1andmore_column"
    }
  }
}
```
- **해결**: `page_info.page_attribute` 경로로 수정, `layout` 필드도 추가

**문제 6: HTML→Markdown 표 변환 separator 중복 버그**

`_html_table_to_markdown()` 함수에서 `lines[-1]`이 방금 추가한 데이터 행을 가리키므로, separator 중복 방지 체크가 의도대로 동작하지 않았음:
```python
# 버그: lines[-1]은 데이터 행이지 separator가 아님
if not (len(lines) >= 2 and re.match(r"^\|[\s\-:|]+\|$", lines[-1])):
    lines.append(sep)
```
- **해결**: 첫 번째 행 뒤에만 separator 1회 추가하는 단순한 로직으로 교체

### 3.4 최종 검증

```
python -m src.eval_parsers --omnidocbench data/omnidocbench/OmniDocBench.json --limit 5
```
```
로드된 페이지: 5
테이블 포함 페이지: 4
총 테이블 수: 4
```

GT Markdown 변환 예시 (중국어 PPT 문서):
```markdown
## 国资背景基金情况
2022年备案基金规模小幅回升，但仍未恢复至资管新规出台前的水平
*2014年-2023Q3国资背景基金的备案数量及规模*
```

GT Markdown 변환 예시 (테이블 포함):
```markdown
| 企业类型 | 目的 | 模式和特点 | 优势 | 典型企业 |
| --- | --- | --- | --- | --- |
| 云服务提供商 | 以物联网为抓手带动上层应用服务业绩增长 | ... | ... | 阿里云、腾讯云 |
```

---

## 4. 핵심 교훈

1. **벤치마크 데이터의 실제 구조를 반드시 먼저 확인해야 한다.** 문서/논문의 스키마 설명과 실제 JSON 구조가 다를 수 있음 (category_type 이름, 필드 위치 등).
2. **`dict.get(key, default)`의 함정**: 키가 존재하되 값이 `None`인 경우 default가 반환되지 않음. `get(key) or default` 패턴을 써야 안전.
3. **1,355개 전체를 보기 전에 통계부터 돌려야 한다.** `Counter`로 category_type 분포를 먼저 확인한 것이 매핑 불일치를 발견하는 데 결정적이었음.
