# 개발 로그: arXiv 기반 테스트 데이터셋 자동 구축

**작업일**: 2026-01-30
**상태**: 파일럿 완료 (8/10 성공), 전체 빌드 대기

---

## 1. 목표

테스트 문서 3개(test_1~3)를 **30개 이상**으로 확장한다. GT(Ground Truth)를 수동 작성하지 않고, arXiv 논문의 **LaTeX 소스 → pandoc → Markdown GT** 자동 변환으로 해결한다.

---

## 2. 사전 분석: Digging (리스크 검토)

구현 전에 계획서를 체계적으로 분석하여 11개 이슈를 발견:

### Critical (3건)
| ID | 이슈 | 대응 |
|----|------|------|
| C-1 | arXiv LaTeX 소스 가용성 과대평가 — 모든 논문에 소스가 있는 것은 아님, rate limiting | 후보 50개 확보, 사전 체크 단계, 3초 딜레이 |
| C-2 | pandoc GT 품질이 "정답" 수준인지 미검증 — 복잡한 표/매크로 변환 실패 가능 | CER < 0.15 기준, 최초 5개 수동 검수 |
| C-3 | 기존 GT 형식과 불일치 — pandoc 출력과 기존 `gt_*.md` 형식 차이 | 후처리에서 형식 정규화 |

### Major (4건)
| ID | 이슈 | 대응 |
|----|------|------|
| M-1 | LaTeX 다중 파일 병합 복잡도 과소평가 | `\input` 재귀 병합, `--resource-path` fallback, `.bbl` 자동 감지 |
| M-2 | 논문 선별 기준에 재현성 관점 부족 | arXiv ID 고정, 선별 근거 필드 추가 |
| M-3 | figure 제거가 Structure F1에 미치는 영향 | figure caption GT 유지 정책 |
| M-4 | pandoc 시스템 의존성 | pandoc CLI 직접 사용 (이미 설치됨), pypandoc fallback |

### Minor (3건)
| ID | 이슈 | 대응 |
|----|------|------|
| m-1 | 폴더 네이밍 `test_arxiv_NNN`이 기존과 불일치 | arXiv 데이터셋 구분 목적으로 유지 |
| m-2 | 30개의 통계적 충분성 근거 없음 | 중심극한정리 n≥30 명시 |
| m-3 | arXiv = 영어만 해당 | "영어 구조 보존 평가용"으로 범위 한정 |

---

## 3. 구현 파일 구조

```
src/dataset/
    __init__.py
    arxiv_paper_list.json      # 50개 후보 논문 (arXiv ID + 메타데이터)
    arxiv_downloader.py        # PDF + LaTeX 소스 다운로드
    latex_to_markdown.py       # LaTeX → pandoc → Markdown 변환
    latex_postprocess.py       # pandoc 출력 후처리
    validate_gt.py             # GT 품질 자동 검증
    build_arxiv_dataset.py     # 전체 파이프라인 오케스트레이터
```

---

## 4. 시행착오 상세

### 4.1 첫 번째 시도: 5개 논문 빌드 → 전멸 (0/5 → 1/5)

**최초 실행 결과**:
```
[1/2] Attention Is All You Need → Conversion failed
[2/2] BERT → Conversion failed
```

**원인 분석**:
- pandoc이 LaTeX 조건문(`\if...\fi`)에서 크래시
- Attention 논문의 에러: `Error at "source" (line 928, column 5): unexpected end of input \fi`

**1차 수정 — `strip_conditionals()` 함수 추가**:
```python
def strip_conditionals(content: str) -> str:
    # \iffalse ... \fi 블록 전체 제거
    content = re.sub(r"\\iffalse\b.*?\\fi\b", "", content, flags=re.DOTALL)
    # 기타 \if 변형 — \if 브랜치만 유지, 5회 반복으로 중첩 처리
    for _ in range(5):
        content = re.sub(r"\\if[a-zA-Z@]*\b[^\n]*\n(.*?)\\else\b(.*?)\\fi\b", r"\1", ...)
        content = re.sub(r"\\if[a-zA-Z@]*\b[^\n]*\n(.*?)\\fi\b", r"\1", ...)
    # 잔여 \fi, \else 제거
    content = re.sub(r"^\\fi\b.*$", "", content, flags=re.MULTILINE)
```

**결과**: Attention 성공 (1/2). BERT 여전히 실패.

---

### 4.2 두 번째 시도: `.bbl` 파일 regex 크래시

**에러**:
```
re.PatternError: bad escape \e at position 28 (line 2, column 1)
```

**원인**: `merge_bbl()` 에서 `.bbl` 파일 내용을 `re.sub`의 replacement로 사용했는데, `.bbl` 파일에 `\entry`, `\end` 같은 LaTeX 명령어가 있어서 regex replacement 특수문자(`\1` 등)로 해석됨.

**수정 전**:
```python
content = re.sub(r"\\bibliography\{[^}]*\}", bbl_content, content)
```

**수정 후** (lambda로 literal replacement):
```python
content = re.sub(r"\\bibliography\{[^}]*\}", lambda _: bbl_content, content)
```

---

### 4.3 세 번째 시도: pandoc이 `\end{abstract}` 거부 (1/5)

**에러 (VGGNet, GPT-3)**:
```
Error at "source" (line 13, column 15):
expecting \end{document}
\end{abstract}
              ^
```

**원인 분석**: `strip_preamble()`이 `\begin{document}` ~ `\end{document}` 사이 본문만 추출한 뒤 pandoc에 넘김. 그런데 본문 안에 `\begin{abstract}...\end{abstract}` 같은 환경이 있고, pandoc은 이것이 **완전한 LaTeX 문서가 아니라고** 거부함.

**시도 1 — 문서 래퍼 추가**:
```python
wrapped = "\\documentclass{article}\n\\begin{document}\n" + content + "\n\\end{document}\n"
```
→ 실패. pandoc이 `\end{abstract}`를 `\end{document}`와 혼동.

**최종 해결 — 전략 2개 파이프라인**:
```python
# Strategy 1: 원본 전체를 pandoc에 넘김 (가장 좋은 품질)
content = merge_inputs(tex_path)
content = merge_bbl(content, tex_path)
content = strip_conditionals(content)      # 크래시 방지만
content = strip_custom_commands(content)   # 크래시 방지만
markdown = convert_with_pandoc(content)    # 전체 문서 그대로

# Strategy 2: Strategy 1 실패 시, 수동 전처리 + 래핑
if markdown is None:
    content = strip_preamble(content)
    content = expand_common_macros(content)
    wrapped = "\\documentclass{article}\\begin{document}" + content + "\\end{document}"
    markdown = convert_with_pandoc(wrapped)
```

**결과**: 3/5 → Attention, BERT, VGGNet 성공.

---

### 4.4 네 번째 시도: `\newcommand`의 `#1`, `#2` 매개변수 (3/5)

**에러 (ResNet)**:
```
Error at "source" (line 209, column 110):
unexpected #2
{*}{\(\left[\begin{array}{c}\text{3$\times$3, #1}\\[-.1em] \text{3$\times$3, #1} \end{array}\right]\)$\times$#2}
```

**원인**: `\newcommand`를 단순 regex로 제거했는데, 중괄호가 여러 줄에 걸쳐 있는 복잡한 정의를 놓침. pandoc이 `#1`, `#2` (매크로 인자)를 만나면 크래시.

**수정 — 중괄호 균형 추적 방식으로 변경**:
```python
def strip_custom_commands(content: str) -> str:
    lines = content.split("\n")
    result = []
    skip_depth = 0
    in_command_def = False

    for line in lines:
        stripped = line.strip()
        if re.match(r"\\(?:new|renew|provide)command|\\def\\|\\DeclareMathOperator", stripped):
            open_b = line.count("{") - line.count("}")
            if open_b <= 0:
                continue  # 한 줄짜리 정의
            else:
                in_command_def = True
                skip_depth = open_b
                continue

        if in_command_def:
            skip_depth += line.count("{") - line.count("}")
            if skip_depth <= 0:
                in_command_def = False
            continue

        result.append(line)
```

**결과**: 4/5 성공. GPT-3도 추가 성공.

---

### 4.5 다섯 번째 시도: `\newcolumntype` (4/5)

**에러 (ResNet)**:
```
Error at "source" (line 284, column 37):
unexpected #1
\newcolumntype{x}[1]{>{\centering}p{#1pt}}
```

**수정**: `strip_custom_commands` 패턴에 `\newcolumntype`, `\newlength`, `\setlength`, `\newcounter`, `\setcounter` 추가.

**결과**: **5/5 전부 성공.**

---

### 4.6 10개 확장 테스트: 8/10 (80%)

```
[1/10] Attention Is All You Need    → OK (48K chars)
[2/10] BERT                         → OK (59K chars, 표 9개)
[3/10] ResNet                       → OK (64K chars)
[4/10] GPT-3                        → OK (279K chars)
[5/10] VGGNet                       → OK (61K chars)
[6/10] Layer Normalization           → FAIL (복잡한 수식 매크로)
[7/10] Adam                         → FAIL (변환 실패)
[8/10] Batch Normalization           → OK (54K chars)
[9/10] GAN                          → OK (63K chars)
[10/10] VAE                         → OK (47K chars)
```

**실패 원인 분석**:
- Layer Norm: `\bm`, `\expectation`, `\fisher` 같은 커스텀 수식 매크로가 대량으로 사용됨. `strip_custom_commands`가 프리앰블의 `\newcommand`는 제거했지만, 본문에 이미 사용된 매크로 호출(`\fisher`)은 남아있어서 pandoc 크래시.
- Adam: 유사한 원인.

**50개 후보 중 80% 성공률 → 약 40개 확보 가능, 목표 30개 초과 달성 예상.**

---

## 5. 발견된 한계

### 5.1 표 감지 부족

10개 논문 중 BERT만 표 9개 감지, 나머지 0개. pandoc이 LaTeX의 복잡한 `tabular` 환경(multirow, multicolumn, cline)을 Markdown pipe table로 변환하지 못하는 경우가 많음.

BERT의 표도 품질 문제가 있었음:
```markdown
| (r0.2cm)1-3 (l0.2cm)4-6 [Mask]{.smallcaps} | [Same]{.smallcaps} | ...
```
→ LaTeX tabular 포맷 잔여물(`(r0.2cm)`, `{.smallcaps}`)이 셀에 남아있음.

### 5.2 CER 기준 전부 FAIL

CER 0.23~0.76 범위로, 0.15 기준 전부 실패. 하지만 이는 **pandoc Markdown vs PyMuPDF raw text** 비교이므로:
- Markdown 포맷팅 (`#`, `**`, `$$`)이 raw text에는 없음
- 참고문헌 형식 차이 (`[bibtex_key]` vs 실제 저자명)
- 수식 표현 차이 (`$h_t$` vs `ht`)

CER 기준을 **정보 참고용**으로 완화하거나, 비교 전에 양쪽 모두 포맷 제거 정규화를 강화해야 함.

---

## 6. 현재 데이터셋 상태

```
data/
    test_1/              # 기존: 한국어 스캔 PDF
    test_2/              # 기존: Chain-of-Thought 논문
    test_3/              # 기존: Attention Is All You Need
    test_arxiv_001/      # Attention Is All You Need
    test_arxiv_002/      # BERT
    test_arxiv_003/      # ResNet
    test_arxiv_004/      # GPT-3
    test_arxiv_005/      # VGGNet
    test_arxiv_006/      # Batch Normalization
    test_arxiv_007/      # GAN
    test_arxiv_008/      # VAE
    _arxiv_raw/          # 다운로드 원본 (PDF + LaTeX 소스)
    omnidocbench/        # OmniDocBench (1,355 페이지)
```

---

## 7. 사용법

### 데이터셋 빌드
```bash
# 전체 50개 후보 빌드
python -m src.dataset.build_arxiv_dataset

# 제한된 수만 빌드
python -m src.dataset.build_arxiv_dataset --limit 10

# 캐시된 다운로드 재사용
python -m src.dataset.build_arxiv_dataset --skip-download

# GT 품질 검증만 실행
python -m src.dataset.validate_gt --dataset-dir data/
```

### 평가 실행
```bash
# 기존 + arXiv 전체 평가 (scan_data_folders가 test_* 자동 발견)
python -m src.eval_parsers --all

# OmniDocBench 평가
python -m src.eval_parsers --omnidocbench data/omnidocbench/OmniDocBench.json --limit 10
```

---

## 8. 다음 단계

1. **나머지 40개 논문 빌드** — `python -m src.dataset.build_arxiv_dataset` 전체 실행
2. **표 감지 개선** — pandoc의 LaTeX tabular 변환 한계를 보완하는 후처리 로직 또는 표가 많은 논문 추가 선별
3. **CER 비교 정규화 강화** — 포맷 제거 후 비교하여 실질적인 텍스트 일치도 측정
4. **수동 GT 검수** — 상위 5개 논문의 GT를 수동으로 확인하여 파이프라인 품질 검증
5. **기존 파서로 test_arxiv_* 평가 실행** — 구조 보존 평가의 통계적 유의성 확보

---

## 9. 핵심 교훈

1. **pandoc은 "깨끗한" LaTeX만 처리 가능하다.** 실제 arXiv 논문은 커스텀 매크로, 조건문, 다중 파일, 복잡한 tabular가 가득하므로 **공격적인 전처리가 필수**.
2. **전략 이중화가 효과적이다.** "원본 그대로 → 실패 시 수동 전처리"의 2단계 전략으로 성공률이 20%→80%로 상승.
3. **중괄호 균형 추적이 regex보다 안정적이다.** `\newcommand` 같은 다중 줄 정의는 단순 regex로 제거 불가. 줄 단위로 `{`, `}` 카운트를 추적하는 방식이 더 robust.
4. **re.sub의 replacement에 raw text를 넣으면 안 된다.** `.bbl` 파일 내용처럼 `\` 가 많은 텍스트는 `lambda _: text` 패턴으로 literal 치환해야 함.
5. **HEAD 요청으로 사전 체크하면 시간을 절약할 수 있다.** arXiv 소스 가용성을 본 다운로드 전에 확인하여 불필요한 다운로드를 방지.
