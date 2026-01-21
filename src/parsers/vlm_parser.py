"""
VLM Parser - Qwen3-VL-2B-Instruct 기반 문서 파싱

Vision-Language Model을 사용하여 이미지에서 구조화된 텍스트를 추출합니다.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  image_bytes                                                │
    │       │                                                     │
    │       ▼                                                     │
    │  _encode_image()  ──────►  image_base64                     │
    │                                 │                           │
    │                                 ▼                           │
    │                           ┌───────────┐                     │
    │                           │  VLM API  │                     │
    │                           │  Request  │                     │
    │                           └───────────┘                     │
    │                                 │                           │
    │                                 ▼                           │
    │                           raw_content (VLM 응답)            │
    │                                 │                           │
    │                                 ▼                           │
    │               _extract_thinking_and_content()               │
    │                                 │                           │
    │                    ┌────────────┴────────────┐              │
    │                    ▼                         ▼              │
    │               thinking                   content            │
    │              (추론 과정)                (실제 결과)           │
    │                    │                         │              │
    │                    └────────────┬────────────┘              │
    │                                 ▼                           │
    │                            VLMResult                        │
    └─────────────────────────────────────────────────────────────┘
"""

import base64
import httpx
import time
from dataclasses import dataclass
from typing import Optional


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class VLMResult:
    """VLM 파싱 결과 데이터 클래스

    Attributes:
        success: API 호출 성공 여부
        content: 추출된 텍스트 (Thinking 제외)
        thinking: Thinking 모델의 추론 과정 (</think> 이전 내용)
        elapsed_time: 처리 소요 시간 (초)
        error: 에러 발생 시 메시지
    """
    success: bool
    content: str
    thinking: Optional[str]
    elapsed_time: float
    error: Optional[str] = None


# ==============================================================================
# Main Class
# ==============================================================================

class VLMParser:
    """Qwen3-VL 기반 문서 파서

    Vision-Language Model API를 호출하여 이미지에서 텍스트를 추출합니다.
    Thinking 모델의 경우 </think> 태그를 기준으로 추론 과정과 결과를 분리합니다.
    HTTP 클라이언트는 Context Manager로 자동 정리됩니다.

    Example:
        >>> parser = VLMParser()
        >>> result = parser.parse(image_bytes)
        >>> print(result.content)  # 마크다운 형식의 텍스트
    """

    # ==========================================================================
    # Class Constants (Prompt for Qwen3-VL)
    # ==========================================================================

    # PROMPT = """
    # You are a document structure extraction expert.
    # Convert this document image to well-structured Markdown.

    # ## Rules:
    # 1. Headers: # for titles, ## for sections, ### for subsections
    # 2. Tables: Markdown table with | separators
    # 3. Lists: - for bullets, 1. for numbered
    # 4. Images/Charts: [Figure: description]
    # 5. Forms: `Field: Value` format

    # ## Important:
    # - Preserve original reading order and hierarchy
    # - Output Markdown only, no explanation

    # ## Output:
    # """

    PROMPT = """
    You are a document TRANSCRIPTION engine, not a writer.
    Your job is to CONVERT the given document image into Markdown by
    STRICTLY TRANSCRIBING what is visible in the image.

    ## Hard Constraints (must follow):
    - DO NOT add, rephrase, summarize, infer, or translate any text.
    - DO NOT explain, comment, or describe what you are doing.
    - If something is partially cut off or unreadable, write `[UNREADABLE]` instead of guessing.
    - If a value is missing in the image, leave it blank or use `[EMPTY]`. Never invent values.

    ## Markdown Formatting Rules:
    1. Headers: use `#` for the main title, `##` for sections, `###` for subsections
    2. Tables: preserve rows/columns as Markdown tables using `|` and `---`
    3. Lists: use `-` for bullets, `1.` for numbered lists
    4. Images/Charts: if there is a visible caption, transcribe it; otherwise use `[Figure]`
    5. Forms: use `Field: Value` exactly as written

    ## Important:
    - Follow the reading order a human would use (left to right, top to bottom).
    - Preserve line breaks and spacing where they matter for meaning (e.g., in tables or forms).
    - Output VALID Markdown ONLY. No extra text before or after.

    ## Output:
    (Write ONLY the transcribed Markdown content here.)
    """

    # ==========================================================================
    # Constructor
    # ==========================================================================

    def __init__(
        self,
        api_url: str = "http://localhost:8005/v1/chat/completions",
        model: str = "qwen3-vl-2b-instruct",
        timeout: float = 120.0
    ):
        """VLMParser 초기화

        Args:
            api_url: VLM API 엔드포인트 URL
            model: 사용할 모델 ID
            timeout: API 요청 타임아웃 (초)
        """
        self.api_url = api_url
        self.model = model
        self.timeout = timeout

    # ==========================================================================
    # Public Methods (Entry Points)
    # ==========================================================================

    def parse(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None
    ) -> VLMResult:
        """이미지에서 구조화된 텍스트 추출 (Entry Point)

        Args:
            image_bytes: 이미지 바이트 데이터 (PNG, JPEG 등)
            prompt: 커스텀 프롬프트 (None이면 PROMPT 사용)

        Returns:
            VLMResult: 파싱 결과 (구조화된 마크다운)

        Flow:
            1. 프롬프트 설정
            2. 이미지 → base64 인코딩
            3. API 페이로드 구성
            4. VLM API 호출
            5. raw_content에서 Thinking/Content 분리
            6. VLMResult 반환
        """
        start_time = time.time()

        # Step 1: 프롬프트 설정 (항상 구조화)
        if prompt is None:
            prompt = self.PROMPT

        # Step 2: 이미지 인코딩
        image_base64 = self._encode_image(image_bytes)

        # Step 3: API 페이로드 구성
        payload = self._build_payload(image_base64, prompt)

        # Step 4-6: API 호출 및 결과 처리 (Context Manager로 자동 리소스 정리)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.api_url, json=payload)
                response.raise_for_status()
                result = response.json()

                # raw_content: VLM API 응답에서 추출한 텍스트
                raw_content = result["choices"][0]["message"]["content"]

                # Step 5: Thinking/Content 분리
                thinking, content = self._extract_thinking_and_content(raw_content)

                return VLMResult(
                    success=True,
                    content=content,
                    thinking=thinking,
                    elapsed_time=time.time() - start_time
                )

        except Exception as e:
            return VLMResult(
                success=False,
                content="",
                thinking=None,
                elapsed_time=time.time() - start_time,
                error=f"이미지 파일을 분석 할 수 없습니다.{str(e)}"
            )

    # ==========================================================================
    # Private Methods (Internal Helpers)
    # ==========================================================================

    def _encode_image(self, image_bytes: bytes) -> str:
        """이미지를 base64 문자열로 인코딩

        Args:
            image_bytes: 원본 이미지 바이트

        Returns:
            base64 인코딩된 문자열
        """
        return base64.b64encode(image_bytes).decode("utf-8")

    def _build_payload(self, image_base64: str, prompt: str) -> dict:
        """VLM API 요청 페이로드 구성 (Qwen3-VL 최적화)

        OpenAI Vision API 호환 형식의 멀티모달 메시지를 구성합니다.
        단일 프롬프트로 역할 지정과 작업 지시를 통합합니다.

        Args:
            image_base64: base64 인코딩된 이미지
            prompt: 통합 프롬프트 (역할 + 규칙 + 출력 형식)

        Returns:
            API 요청 페이로드 딕셔너리
        """
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1
        }

    def _extract_thinking_and_content(
        self,
        raw_content: str
    ) -> tuple[Optional[str], str]:
        """
        Thinking 모델 출력에서 추론 과정과 실제 결과 분리

        Args:
            raw_content: VLM API의 원본 응답 텍스트

        Returns:
            (thinking, content) 튜플
            - thinking: </think> 이전 내용, 없으면 None
            - content: </think> 이후 내용 (실제 결과)
        """
        if "</think>" in raw_content:
            parts = raw_content.split("</think>", 1)
            thinking = parts[0].replace("<think>", "").strip()
            content = parts[1].strip() if len(parts) > 1 else ""
            return thinking, content

        return None, raw_content
