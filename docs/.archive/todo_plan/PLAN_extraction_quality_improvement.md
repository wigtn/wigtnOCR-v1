# Task Plan: Fair CER Evaluation Pipeline

> **Generated from**: docs/prd/PRD_extraction_quality_improvement.md v2.0
> **Created**: 2026-01-30
> **Status**: pending

## Execution Config

| Option | Value | Description |
|--------|-------|-------------|
| `auto_commit` | true | 완료 시 자동 커밋 |
| `commit_per_phase` | false | Phase별 중간 커밋 여부 |
| `quality_gate` | true | /auto-commit 품질 검사 |

## Phases

### Phase 1: 공정 정규화 완성
- [ ] FR-001: normalize_text()에 숫자 인용 참조 `[1]`, `[1,2]`, `[1-3]` 제거 추가
- [ ] FR-002: 수평선 `---` 제거 추가
- [ ] FR-006: 각주 정의 `[^N]: text` 줄 전체 제거 추가
- [ ] FR-005: 괄호 인용 `(Author et al., 2020)` 제거 추가

### Phase 2: 섹션별 CER + 상세 저장
- [ ] FR-003: `split_body_references()` 함수 구현
- [ ] FR-003: `calculate_body_cer()` 함수 구현
- [ ] FR-003: `evaluate_results()`에서 Body CER 계산 및 출력
- [ ] FR-003: `save_results_to_files()`에서 Body CER 저장
- [ ] FR-004: `save_results_to_files()`에서 cer_detail (S/D/I/hits) 저장

### Phase 3: 검증
- [ ] 39개 arXiv 데이터셋 재평가 실행
- [ ] Full CER vs Body CER 비교 분석
- [ ] Body CER < Full CER 정합성 확인

## Progress

| Metric | Value |
|--------|-------|
| Total Tasks | 0/12 |
| Current Phase | - |
| Status | pending |

## Execution Log

| Timestamp | Phase | Task | Status |
|-----------|-------|------|--------|
| - | - | - | - |
