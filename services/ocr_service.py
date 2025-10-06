import time
import base64
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from app.deps import get_gemini_llm
from models.metrics import OCRMetrics, get_ocr_metrics_aggregator

logger = logging.getLogger(__name__)

system_prompt = """
    너는 이미지를 OCR로 읽고 핵심만 한국어로 요약하는 도우미다.
    민감정보/좌표/박스/메타데이터는 절대 포함하지 마라.
    응답은 마크다운을 사용하지 말고, 순수 텍스트만 포함해라.
"""
user_prompt = """
    아래 이미지를 OCR하고, 핵심만 요약해 주세요.
    - 최대 1500자 이내
    - 불필요한 수식어/중복 제거
    - 표/레이아웃/좌표/메타데이터 금지
    - 중요도 순 5~10개 불릿
    - 너무 길어질 경우 상위 중요 정보만
"""


async def extract_text_from_image(img_base64: str) -> str:
    img_bytes = base64.b64decode(img_base64)
    image_size_kb = len(img_bytes) / 1024

    model = get_gemini_llm()

    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": user_prompt
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            }
        ]
    )

    logger.info(f"Gemini OCR 요청 시작 - 이미지 크기: {image_size_kb:.2f}KB")

    start_time = time.time()
    response = await model.ainvoke([system_message, human_message])
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000

    # 응답 메타데이터에서 토큰 사용량 추출
    input_tokens = getattr(response, 'usage_metadata', {}).get('input_tokens', 0) if hasattr(response, 'usage_metadata') else 0
    output_tokens = getattr(response, 'usage_metadata', {}).get('output_tokens', 0) if hasattr(response, 'usage_metadata') else 0
    total_tokens = input_tokens + output_tokens

    result_text = response.content if response.content else ""

    # 메트릭 생성
    metrics = OCRMetrics(
        duration_ms=duration_ms,
        image_size_kb=image_size_kb,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        result_length=len(result_text),
        success=True
    )

    # 전역 집계기에 메트릭 추가
    aggregator = get_ocr_metrics_aggregator()
    aggregator.add_metrics(metrics)

    return result_text
