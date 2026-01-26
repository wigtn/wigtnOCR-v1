#!/usr/bin/env python3
"""
CLI 기반 Parser 테스트 스크립트

다양한 입력 포맷을 지원하는 Parser 비교 테스트:
1. VLM Parser (Qwen3-VL-2B-Instruct) - 모든 포맷 지원
2. OCR Parser - Text (pdfplumber) - PDF 전용
3. OCR Parser - Image (Docling + RapidOCR) - PDF 전용

지원 포맷:
- PDF: 디지털/스캔 PDF (모든 파서 테스트)
- IMAGE: PNG, JPG, JPEG (VLM만 지원)
- HWP/HWPX: 한글 문서 (이미지로 변환 후 VLM 파싱)

Usage:
    # PDF 테스트
    python -m src.test_parsers --input data/sample.pdf --gt data/ground_truth.md

    # 이미지 테스트 (VLM만)
    python -m src.test_parsers --input data/image.png --gt data/ground_truth.md

    # HWP/HWPX 테스트 (VLM만, LibreOffice 필요)
    python -m src.test_parsers --input data/document.hwp --gt data/ground_truth.md

    # 레거시 (--pdf 옵션도 지원)
    python -m src.test_parsers --pdf data/sample.pdf
"""

import argparse
import subprocess
import sys
import time
import re
import tempfile
from pathlib import Path
from enum import Enum
from typing import Optional, List

import jiwer

# =============================================================================
# Import Compatibility Layer
# =============================================================================
# 두 가지 실행 방식 모두 지원:
#   1. python -m src.test_parsers (프로젝트 루트에서)
#   2. python test_parsers.py (src/ 디렉토리에서)

def _import_parsers():
    """파서 모듈을 동적으로 임포트 (경로 호환성 처리)"""
    global VLMParser, OCRParser, ImageOCRParser, DoclingParser, check_docling_available

    try:
        # 방법 1: src.parsers (프로젝트 루트에서 실행)
        from src.parsers.vlm_parser import VLMParser
        from src.parsers.ocr_parser import OCRParser, ImageOCRParser
        from src.parsers.docling_parser import DoclingParser, check_docling_available
    except ImportError:
        try:
            # 방법 2: parsers (src/ 디렉토리에서 실행 또는 PYTHONPATH 설정)
            from parsers.vlm_parser import VLMParser
            from parsers.ocr_parser import OCRParser, ImageOCRParser
            from parsers.docling_parser import DoclingParser, check_docling_available
        except ImportError as e:
            print(f"❌ 파서 모듈을 찾을 수 없습니다: {e}")
            print("   프로젝트 루트에서 실행하세요: python -m src.test_parsers")
            sys.exit(1)

    return VLMParser, OCRParser, ImageOCRParser, DoclingParser, check_docling_available

# 지연 임포트를 위한 플레이스홀더
VLMParser = None
OCRParser = None
ImageOCRParser = None
DoclingParser = None
check_docling_available = None


# =============================================================================
# File Format Detection
# =============================================================================

class FileFormat(Enum):
    """지원하는 파일 포맷"""
    PDF = "pdf"
    IMAGE = "image"  # PNG, JPG, JPEG
    HWP = "hwp"      # 한글 97-2003
    HWPX = "hwpx"    # 한글 2010+
    UNKNOWN = "unknown"


def detect_file_format(file_path: Path) -> FileFormat:
    """파일 확장자로 포맷 감지"""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return FileFormat.PDF
    elif suffix in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"]:
        return FileFormat.IMAGE
    elif suffix == ".hwp":
        return FileFormat.HWP
    elif suffix == ".hwpx":
        return FileFormat.HWPX
    else:
        return FileFormat.UNKNOWN


def convert_hwp_to_images(hwp_path: Path, dpi: int = 150) -> List[bytes]:
    """HWP/HWPX 파일을 이미지로 변환 (LibreOffice 사용)

    Args:
        hwp_path: HWP/HWPX 파일 경로
        dpi: 출력 해상도

    Returns:
        PNG 이미지 바이트 리스트

    Requirements:
        - LibreOffice 설치 필요 (soffice 명령)
        - Ubuntu: sudo apt install libreoffice
    """
    images = []

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Step 1: HWP → PDF 변환 (LibreOffice)
            print("   HWP → PDF 변환 중 (LibreOffice)...", end=" ", flush=True)
            result = subprocess.run([
                "soffice",
                "--headless",
                "--convert-to", "pdf",
                "--outdir", str(tmpdir_path),
                str(hwp_path)
            ], capture_output=True, timeout=60)

            if result.returncode != 0:
                print("✗")
                print(f"   LibreOffice 오류: {result.stderr.decode()}")
                return []

            print("✓")

            # Step 2: 변환된 PDF 찾기
            pdf_files = list(tmpdir_path.glob("*.pdf"))
            if not pdf_files:
                print("   ❌ PDF 파일 생성 실패")
                return []

            pdf_path = pdf_files[0]
            pdf_bytes = pdf_path.read_bytes()

            # Step 3: PDF → 이미지 변환
            global ImageOCRParser
            if ImageOCRParser is None:
                _import_parsers()
            image_parser = ImageOCRParser()
            images = image_parser.pdf_to_images(pdf_bytes, dpi=dpi)

            print(f"   PDF → {len(images)} 페이지 이미지 변환 완료")

    except FileNotFoundError:
        print("\n   ❌ LibreOffice가 설치되지 않았습니다.")
        print("   Ubuntu: sudo apt install libreoffice")
        print("   macOS: brew install --cask libreoffice")
    except subprocess.TimeoutExpired:
        print("\n   ❌ 변환 타임아웃 (60초 초과)")
    except Exception as e:
        print(f"\n   ❌ HWP 변환 오류: {e}")

    return images


# =============================================================================
# Text Normalization (for fair comparison)
# =============================================================================

def normalize_text(text: str) -> str:
    """텍스트 정규화 - CER/WER 비교를 위한 전처리

    - 마크다운 문법 제거
    - 다중 공백/줄바꿈 정규화
    - 앞뒤 공백 제거
    """
    if not text:
        return ""

    result = text

    # 마크다운 헤더 제거
    result = re.sub(r'^#{1,6}\s+', '', result, flags=re.MULTILINE)

    # Bold/Italic 제거
    result = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', result)
    result = re.sub(r'\*\*(.+?)\*\*', r'\1', result)
    result = re.sub(r'\*(.+?)\*', r'\1', result)
    result = re.sub(r'___(.+?)___', r'\1', result)
    result = re.sub(r'__(.+?)__', r'\1', result)
    result = re.sub(r'_(.+?)_', r'\1', result)

    # 링크 제거
    result = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', result)

    # 리스트 마커 제거
    result = re.sub(r'^[\s]*[-*+]\s+', '', result, flags=re.MULTILINE)
    result = re.sub(r'^[\s]*\d+\.\s+', '', result, flags=re.MULTILINE)

    # 테이블 파이프 제거
    result = re.sub(r'\|', ' ', result)

    # 다중 공백 정규화
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r'\n{2,}', '\n', result)

    # 각 줄 앞뒤 공백 제거
    lines = [line.strip() for line in result.split('\n')]
    lines = [line for line in lines if line]

    return '\n'.join(lines).strip()


# =============================================================================
# Tokenizer
# =============================================================================

def get_tokenizer(tokenizer_type: str = "whitespace"):
    """토크나이저 반환

    Args:
        tokenizer_type: "whitespace", "mecab", "okt"

    Returns:
        tokenize 함수
    """
    if tokenizer_type == "mecab":
        try:
            from konlpy.tag import Mecab
            mecab = Mecab()
            print("✓ Mecab 토크나이저 사용")
            return mecab.morphs
        except ImportError:
            print("⚠️ konlpy가 설치되지 않았습니다. whitespace로 fallback")
            return lambda x: x.split()
        except Exception as e:
            print(f"⚠️ Mecab 초기화 실패: {e}. whitespace로 fallback")
            return lambda x: x.split()

    elif tokenizer_type == "okt":
        try:
            from konlpy.tag import Okt
            okt = Okt()
            print("✓ Okt 토크나이저 사용 (순수 Python, 시스템 의존성 없음)")
            return okt.morphs
        except ImportError:
            print("⚠️ konlpy가 설치되지 않았습니다. whitespace로 fallback")
            return lambda x: x.split()
        except Exception as e:
            print(f"⚠️ Okt 초기화 실패: {e}. whitespace로 fallback")
            return lambda x: x.split()

    else:
        return lambda x: x.split()


# =============================================================================
# CER/WER Calculation (using jiwer library)
# =============================================================================


def calculate_cer(hypothesis: str, reference: str) -> dict:
    """Character Error Rate 계산 (jiwer 사용)

    Returns:
        dict with cer, substitutions, deletions, insertions
    """
    if not reference:
        return {"cer": 0.0 if not hypothesis else float('inf')}

    if not hypothesis:
        return {
            "cer": 1.0,
            "substitutions": 0,
            "deletions": len(reference),
            "insertions": 0
        }

    # jiwer.cer: reference first, hypothesis second
    cer_value = jiwer.cer(reference, hypothesis)

    # 상세 정보
    output = jiwer.process_characters(reference, hypothesis)

    return {
        "cer": cer_value,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits
    }


def calculate_wer(hypothesis: str, reference: str, tokenizer=None) -> dict:
    """Word Error Rate 계산 (jiwer 사용)

    Args:
        hypothesis: 파서 출력 텍스트
        reference: Ground Truth 텍스트
        tokenizer: 토크나이저 함수 (None이면 공백 분리)

    Returns:
        dict with wer, substitutions, deletions, insertions
    """
    def default_tokenizer(x: str) -> list[str]:
        return x.split()

    if tokenizer is None:
        tokenizer = default_tokenizer

    if not reference:
        return {"wer": 0.0 if not hypothesis else float('inf')}

    # 토큰화
    ref_tokens = tokenizer(reference)
    hyp_tokens = tokenizer(hypothesis) if hypothesis else []

    if not hyp_tokens:
        return {
            "wer": 1.0,
            "substitutions": 0,
            "deletions": len(ref_tokens),
            "insertions": 0,
            "ref_tokens": len(ref_tokens),
            "hyp_tokens": 0
        }

    # 토큰을 공백으로 연결하여 jiwer에 전달
    ref_str = " ".join(ref_tokens)
    hyp_str = " ".join(hyp_tokens)

    # jiwer.wer: reference first, hypothesis second
    wer_value = jiwer.wer(ref_str, hyp_str)

    # 상세 정보
    output = jiwer.process_words(ref_str, hyp_str)

    return {
        "wer": wer_value,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
        "ref_tokens": len(ref_tokens),
        "hyp_tokens": len(hyp_tokens)
    }


# =============================================================================
# Parser Tests
# =============================================================================

def test_vlm_parser(
    input_data: bytes,
    verbose: bool = False,
    file_format: FileFormat = FileFormat.PDF,
    pre_converted_images: Optional[List[bytes]] = None
) -> dict:
    """VLM Parser 테스트 (다양한 입력 포맷 지원)

    Args:
        input_data: 파일 바이트 데이터
        verbose: 상세 출력 여부
        file_format: 파일 포맷
        pre_converted_images: 미리 변환된 이미지 리스트 (HWP용)
    """
    global VLMParser, ImageOCRParser
    if VLMParser is None:
        _import_parsers()

    print("\n" + "=" * 60)
    print("🤖 VLM Parser (Qwen3-VL-2B-Instruct)")
    print("=" * 60)

    start_time = time.time()

    # 입력 포맷에 따른 이미지 준비
    if pre_converted_images:
        # HWP/HWPX: 미리 변환된 이미지 사용
        images = pre_converted_images
        print(f"📄 입력: 미리 변환된 이미지 ({len(images)} 페이지)")

    elif file_format == FileFormat.IMAGE:
        # 이미지 파일: 바로 사용
        images = [input_data]
        print("📄 입력: 단일 이미지 파일")

    elif file_format == FileFormat.PDF:
        # PDF → 이미지 변환
        image_parser = ImageOCRParser()
        images = image_parser.pdf_to_images(input_data, dpi=150)

        if not images:
            print("❌ PDF → 이미지 변환 실패")
            return {"success": False, "error": "이미지 변환 실패"}

        print(f"📄 페이지 수: {len(images)}")

    else:
        print(f"❌ 지원하지 않는 포맷: {file_format}")
        return {"success": False, "error": f"Unsupported format: {file_format}"}

    # VLM 파싱
    vlm_parser = VLMParser()
    results = []

    for i, img_bytes in enumerate(images):
        print(f"  처리 중: Page {i+1}/{len(images)}...", end=" ", flush=True)
        result = vlm_parser.parse(img_bytes)
        results.append(result)

        if result.success:
            print(f"✓ ({result.elapsed_time:.2f}s)")
        else:
            print(f"✗ ({result.error})")

    total_time = time.time() - start_time

    # 결과 합치기
    combined_content = "\n\n".join(
        r.content for r in results if r.success and r.content
    )

    success_count = sum(1 for r in results if r.success)

    print("\n📊 결과:")
    print(f"   - 성공: {success_count}/{len(results)} 페이지")
    print(f"   - 총 시간: {total_time:.2f}s")
    print(f"   - 평균: {total_time/len(images):.2f}s/page")
    print(f"   - 추출 길이: {len(combined_content)} chars")

    if verbose and combined_content:
        print("\n📝 추출 결과 (처음 500자):")
        print("-" * 40)
        print(combined_content[:500])
        print("-" * 40)

    return {
        "success": success_count > 0,
        "content": combined_content,
        "elapsed_time": total_time,
        "page_count": len(images),
        "per_page_time": total_time / len(images)
    }


def test_ocr_text_parser(pdf_bytes: bytes, verbose: bool = False) -> dict:
    """OCR Parser (Text - pdfplumber) 테스트"""
    global OCRParser
    if OCRParser is None:
        _import_parsers()

    print("\n" + "=" * 60)
    print("📖 OCR Parser - Text (pdfplumber)")
    print("=" * 60)

    parser = OCRParser()

    # PDF 타입 확인
    pdf_type = parser.detect_pdf_type(pdf_bytes)
    print(f"📄 PDF 타입: {pdf_type}")

    # 파싱
    result = parser.parse_pdf(pdf_bytes)

    print("\n📊 결과:")
    print(f"   - 성공: {'✓' if result.success else '✗'}")
    print(f"   - 페이지 수: {result.page_count}")
    print(f"   - 텍스트 존재: {'✓' if result.has_text else '✗'}")
    print(f"   - 표 개수: {len(result.tables)}")
    print(f"   - 총 시간: {result.elapsed_time:.2f}s")
    print(f"   - 추출 길이: {len(result.content)} chars")

    if verbose and result.content:
        print("\n📝 추출 결과 (처음 500자):")
        print("-" * 40)
        print(result.content[:500])
        print("-" * 40)

    return {
        "success": result.success,
        "content": result.content,
        "elapsed_time": result.elapsed_time,
        "page_count": result.page_count,
        "has_text": result.has_text,
        "pdf_type": pdf_type
    }


def test_ocr_image_parser(pdf_bytes: bytes, verbose: bool = False) -> dict:
    """OCR Parser (Image - Docling) 테스트"""
    global DoclingParser, check_docling_available
    if DoclingParser is None:
        _import_parsers()

    print("\n" + "=" * 60)
    print("🔍 OCR Parser - Image (Docling + RapidOCR)")
    print("=" * 60)

    if not check_docling_available():
        print("❌ Docling 라이브러리가 설치되지 않았습니다.")
        print("   pip install docling 명령으로 설치하세요.")
        return {"success": False, "error": "Docling not installed"}

    parser = DoclingParser(ocr_enabled=True)
    result = parser.parse_pdf(pdf_bytes)

    print("\n📊 결과:")
    print(f"   - 성공: {'✓' if result.success else '✗'}")
    print(f"   - 페이지 수: {result.page_count}")
    print(f"   - 총 시간: {result.elapsed_time:.2f}s")
    print(f"   - 추출 길이 (text): {len(result.content)} chars")
    print(f"   - 추출 길이 (markdown): {len(result.markdown)} chars")

    if result.error:
        print(f"   - 에러: {result.error}")

    if verbose and result.content:
        print("\n📝 추출 결과 (처음 500자):")
        print("-" * 40)
        print(result.content[:500])
        print("-" * 40)

    return {
        "success": result.success,
        "content": result.content,
        "markdown": result.markdown,
        "elapsed_time": result.elapsed_time,
        "page_count": result.page_count,
        "error": result.error
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_results(results: dict, ground_truth: str, tokenizer=None, tokenizer_name: str = "whitespace") -> dict:
    """결과 평가 (CER, WER 계산) - jiwer 사용

    Args:
        results: 파서별 결과 딕셔너리
        ground_truth: Ground Truth 텍스트
        tokenizer: WER 계산용 토크나이저 함수
        tokenizer_name: 토크나이저 이름 (출력용)
    """
    print("\n" + "=" * 60)
    print("📊 평가 결과 (Ground Truth 비교) - jiwer")
    print(f"   WER Tokenizer: {tokenizer_name}")
    print("=" * 60)

    gt_normalized = normalize_text(ground_truth)
    print(f"Ground Truth 길이: {len(gt_normalized)} chars (정규화 후)")
    print()

    evaluation = {}

    for parser_name, result in results.items():
        if not result.get("success"):
            print(f"{parser_name}: SKIP (파싱 실패)")
            evaluation[parser_name] = {"cer": None, "wer": None}
            continue

        content = result.get("content", "")
        if not content:
            print(f"{parser_name}: SKIP (내용 없음)")
            evaluation[parser_name] = {"cer": None, "wer": None}
            continue

        # 정규화
        content_normalized = normalize_text(content)

        # CER, WER 계산 (jiwer)
        cer_result = calculate_cer(content_normalized, gt_normalized)
        wer_result = calculate_wer(content_normalized, gt_normalized, tokenizer)

        cer = cer_result["cer"]
        wer = wer_result["wer"]

        print(f"{parser_name}:")
        print(f"   - CER: {cer:.4f} ({cer*100:.2f}%)")
        print(f"      └─ S:{cer_result.get('substitutions', 0)} D:{cer_result.get('deletions', 0)} I:{cer_result.get('insertions', 0)}")
        print(f"   - WER: {wer:.4f} ({wer*100:.2f}%)")
        print(f"      └─ S:{wer_result.get('substitutions', 0)} D:{wer_result.get('deletions', 0)} I:{wer_result.get('insertions', 0)}")
        print(f"      └─ Tokens: ref={wer_result.get('ref_tokens', 0)} hyp={wer_result.get('hyp_tokens', 0)}")
        print(f"   - Latency: {result.get('elapsed_time', 0):.2f}s")
        print()

        evaluation[parser_name] = {
            "cer": cer,
            "cer_detail": cer_result,
            "wer": wer,
            "wer_detail": wer_result,
            "latency": result.get("elapsed_time", 0),
            "tokenizer": tokenizer_name
        }

    return evaluation


def save_results_to_files(results: dict, output_dir: str, pdf_name: str, evaluation: dict = None, tokenizer_name: str = "whitespace"):
    """파싱 결과를 파일로 저장"""
    from pathlib import Path
    import json
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _ = Path(pdf_name).stem  # base_name reserved for future use

    print("\n" + "=" * 60)
    print(f"💾 결과 저장: {output_path}")
    print("=" * 60)

    saved_files = []

    # 1. 각 파서별 출력 저장
    for parser_name, result in results.items():
        if not result.get("success"):
            continue

        content = result.get("content", "")
        if not content:
            continue

        safe_name = parser_name.lower().replace(" ", "-")
        ext = ".md" if "markdown" in result else ".txt"
        filename = f"{safe_name}_output{ext}"
        filepath = output_path / filename

        save_content = result.get("markdown", content)
        filepath.write_text(save_content, encoding="utf-8")
        saved_files.append(filepath)
        print(f"   ✓ {filename} ({len(save_content)} chars)")

    # 2. 평가 결과 JSON 저장
    meta = {
        "pdf": pdf_name,
        "timestamp": timestamp,
        "tokenizer": tokenizer_name,
        "results": {}
    }
    for name, result in results.items():
        meta["results"][name] = {
            "success": result.get("success"),
            "elapsed_time": result.get("elapsed_time"),
            "content_length": len(result.get("content", ""))
        }
        if evaluation and name in evaluation:
            eval_data = evaluation[name]
            meta["results"][name]["cer"] = eval_data.get("cer")
            meta["results"][name]["wer"] = eval_data.get("wer")

    meta_path = output_path / "evaluation.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("   ✓ evaluation.json")

    # 3. 요약 마크다운 저장
    summary_lines = [
        "# Parsing Test Results",
        "",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **PDF**: {pdf_name}",
        f"- **Tokenizer**: {tokenizer_name}",
        "",
        "## Results",
        "",
        "| Parser | Success | Latency | Length |",
        "|--------|---------|---------|--------|",
    ]

    for name, result in results.items():
        success = "✓" if result.get("success") else "✗"
        latency = f"{result.get('elapsed_time', 0):.2f}s"
        length = f"{len(result.get('content', ''))} chars"
        summary_lines.append(f"| {name} | {success} | {latency} | {length} |")

    if evaluation:
        summary_lines.extend([
            "",
            "## Evaluation (vs Ground Truth)",
            "",
            "| Parser | CER | WER |",
            "|--------|-----|-----|",
        ])
        for name, eval_data in evaluation.items():
            cer = eval_data.get("cer")
            wer = eval_data.get("wer")
            cer_str = f"{cer*100:.2f}%" if cer is not None else "N/A"
            wer_str = f"{wer*100:.2f}%" if wer is not None else "N/A"
            summary_lines.append(f"| {name} | {cer_str} | {wer_str} |")

    summary_path = output_path / "README.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("   ✓ README.md (요약)")

    return saved_files


def print_summary(results: dict, evaluation: dict = None):
    """결과 요약 출력"""
    print("\n" + "=" * 60)
    print("📋 테스트 요약")
    print("=" * 60)

    print("\n| Parser | 성공 | 시간 | 추출 길이 |")
    print("|--------|------|------|----------|")

    for name, result in results.items():
        success = "✓" if result.get("success") else "✗"
        time_str = f"{result.get('elapsed_time', 0):.2f}s"
        length = len(result.get("content", ""))
        print(f"| {name} | {success} | {time_str} | {length} chars |")

    if evaluation:
        print("\n| Parser | CER | WER | Latency |")
        print("|--------|-----|-----|---------|")

        for name, eval_result in evaluation.items():
            cer = eval_result.get("cer")
            wer = eval_result.get("wer")
            latency = eval_result.get("latency", 0)

            cer_str = f"{cer*100:.2f}%" if cer is not None else "N/A"
            wer_str = f"{wer*100:.2f}%" if wer is not None else "N/A"

            print(f"| {name} | {cer_str} | {wer_str} | {latency:.2f}s |")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="다중 포맷 Parser 비교 테스트 (VLM, OCR-Text, OCR-Image)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
지원 포맷:
  PDF       : 디지털/스캔 PDF (모든 파서 테스트)
  IMAGE     : PNG, JPG, JPEG, WebP, GIF, BMP, TIFF (VLM만)
  HWP/HWPX  : 한글 문서 (LibreOffice로 변환 후 VLM)

예시:
  python -m src.test_parsers --input data/test.pdf --gt data/gt.md
  python -m src.test_parsers --input data/receipt.png --gt data/gt.md
  python -m src.test_parsers --input data/document.hwp --gt data/gt.md
        """
    )

    # 입력 파일 (--input 또는 레거시 --pdf)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        help="테스트할 입력 파일 (PDF, 이미지, HWP/HWPX)"
    )
    input_group.add_argument(
        "--pdf", "-p",
        help="테스트할 PDF 파일 경로 (레거시 옵션, --input 사용 권장)"
    )

    parser.add_argument(
        "--gt", "-g",
        help="Ground Truth 파일 경로 (선택)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 출력 (추출 결과 미리보기)"
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="VLM Parser 테스트 스킵"
    )
    parser.add_argument(
        "--skip-docling",
        action="store_true",
        help="Docling Parser 테스트 스킵"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="파싱 결과를 저장할 디렉토리"
    )
    parser.add_argument(
        "--save-docs",
        action="store_true",
        help="결과를 docs/Parsing_test_<일자>/ 폴더에 저장"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        choices=["whitespace", "mecab", "okt"],
        default="whitespace",
        help="WER 계산용 토크나이저 (기본: whitespace, 한국어: mecab 또는 okt)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PDF/HWP → 이미지 변환 해상도 (기본: 150)"
    )

    args = parser.parse_args()

    # --save-docs 옵션 처리
    if args.save_docs:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        args.output_dir = f"docs/Parsing_test_{date_str}"

    # 입력 파일 경로 결정 (--input 우선, --pdf는 레거시)
    input_path = Path(args.input if args.input else args.pdf)
    if not input_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    # 파일 포맷 감지
    file_format = detect_file_format(input_path)

    print("=" * 60)
    print("🔬 VLM Document Parsing Quality Test")
    print("=" * 60)
    print(f"📄 입력 파일: {input_path}")
    print(f"📁 포맷: {file_format.value.upper()}")

    input_bytes = input_path.read_bytes()
    print(f"📦 크기: {len(input_bytes) / 1024:.1f} KB")

    # 지원하지 않는 포맷 체크
    if file_format == FileFormat.UNKNOWN:
        print("❌ 지원하지 않는 파일 포맷입니다.")
        print("   지원 포맷: PDF, PNG, JPG, JPEG, HWP, HWPX")
        sys.exit(1)

    # Ground Truth 읽기 (선택)
    ground_truth = None
    if args.gt:
        gt_path = Path(args.gt)
        if gt_path.exists():
            ground_truth = gt_path.read_text(encoding="utf-8")
            print(f"📋 Ground Truth: {gt_path} ({len(ground_truth)} chars)")
        else:
            print(f"⚠️ Ground Truth 파일을 찾을 수 없습니다: {gt_path}")

    results = {}

    # HWP/HWPX 전처리 (이미지로 변환)
    hwp_images = None
    if file_format in [FileFormat.HWP, FileFormat.HWPX]:
        print("\n📄 HWP/HWPX 변환 시작...")
        hwp_images = convert_hwp_to_images(input_path, dpi=args.dpi)

        if not hwp_images:
            print("❌ HWP → 이미지 변환 실패. LibreOffice 설치를 확인하세요.")
            sys.exit(1)

    # ==========================================================================
    # 포맷별 테스트 분기
    # ==========================================================================

    if file_format == FileFormat.PDF:
        # PDF: 모든 파서 테스트

        # 1. VLM Parser
        if not args.skip_vlm:
            try:
                results["VLM"] = test_vlm_parser(
                    input_bytes, args.verbose, FileFormat.PDF
                )
            except Exception as e:
                print(f"❌ VLM Parser 오류: {e}")
                results["VLM"] = {"success": False, "error": str(e)}

        # 2. OCR Parser (Text)
        try:
            results["OCR-Text"] = test_ocr_text_parser(input_bytes, args.verbose)
        except Exception as e:
            print(f"❌ OCR-Text Parser 오류: {e}")
            results["OCR-Text"] = {"success": False, "error": str(e)}

        # 3. OCR Parser (Image - Docling)
        if not args.skip_docling:
            try:
                results["OCR-Image"] = test_ocr_image_parser(input_bytes, args.verbose)
            except Exception as e:
                print(f"❌ OCR-Image Parser 오류: {e}")
                results["OCR-Image"] = {"success": False, "error": str(e)}

    elif file_format == FileFormat.IMAGE:
        # 이미지: VLM만 테스트
        print("\n⚠️ 이미지 입력: VLM Parser만 테스트합니다 (OCR 파서는 PDF 전용)")

        if args.skip_vlm:
            print("❌ --skip-vlm 옵션이 설정되어 테스트할 파서가 없습니다.")
            sys.exit(1)

        try:
            results["VLM"] = test_vlm_parser(
                input_bytes, args.verbose, FileFormat.IMAGE
            )
        except Exception as e:
            print(f"❌ VLM Parser 오류: {e}")
            results["VLM"] = {"success": False, "error": str(e)}

    elif file_format in [FileFormat.HWP, FileFormat.HWPX]:
        # HWP/HWPX: VLM만 테스트 (이미지로 변환됨)
        print("\n⚠️ HWP/HWPX 입력: VLM Parser만 테스트합니다")

        if args.skip_vlm:
            print("❌ --skip-vlm 옵션이 설정되어 테스트할 파서가 없습니다.")
            sys.exit(1)

        try:
            results["VLM"] = test_vlm_parser(
                input_bytes, args.verbose, file_format,
                pre_converted_images=hwp_images
            )
        except Exception as e:
            print(f"❌ VLM Parser 오류: {e}")
            results["VLM"] = {"success": False, "error": str(e)}

    # 평가 (Ground Truth가 있는 경우)
    evaluation = None
    if ground_truth:
        tokenizer = get_tokenizer(args.tokenizer)
        evaluation = evaluate_results(results, ground_truth, tokenizer, args.tokenizer)

    # 결과 파일 저장 (--output-dir 또는 --save-docs 옵션)
    if args.output_dir:
        save_results_to_files(results, args.output_dir, str(input_path), evaluation, args.tokenizer)

    # 요약 출력
    print_summary(results, evaluation)


if __name__ == "__main__":
    main()
