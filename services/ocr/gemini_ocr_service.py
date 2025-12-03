import base64
import asyncio
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import (
  retry,
  stop_after_attempt,
  wait_exponential,
  retry_if_exception_type,
  before_sleep_log
)
from app.deps import get_gemini_llm
from services.ocr.base import BaseOCRService
from langchain_core.callbacks import UsageMetadataCallbackHandler

logger = logging.getLogger(__name__)

class GeminiOCRService(BaseOCRService):
    def __init__(self):
        self.system_prompt = """
            너는 이미지를 OCR로 읽고 핵심만 한국어로 요약하는 도우미다.
            민감정보/좌표/박스/메타데이터는 절대 포함하지 마라.
            응답은 마크다운을 사용하지 말고, 순수 텍스트만 포함해라.
        """
        self.user_prompt = """
            아래 이미지를 OCR하고, 핵심만 요약해 주세요.
            - 최대 1500자 이내
            - 불필요한 수식어/중복 제거
            - 표/레이아웃/좌표/메타데이터 금지
        """

    @retry(
        stop=stop_after_attempt(3),  # 최대 3회 시도
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 2초, 4초, 8초 대기
        retry=retry_if_exception_type((TimeoutError, asyncio.TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def extract_text_from_image(self, img_base64: str) -> str:
        img_bytes = base64.b64decode(img_base64)
        image_size_kb = len(img_bytes) / 1024

        model = get_gemini_llm()

        system_message = SystemMessage(content=self.system_prompt)
        human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": self.user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                }
            ]
        )

        logger.info(f"Gemini OCR 요청 시작 - 이미지 크기: {image_size_kb:.2f}KB")

        usage_callback = UsageMetadataCallbackHandler()

        try:
            response = await model.ainvoke(
                [system_message, human_message],
                config={"callbacks": [usage_callback]}
            )
        except Exception as e:
            logger.error(f"OCR 요청 실패 - 이미지 크기: {image_size_kb:.2f}KB, 에러: {type(e).__name__}: {str(e)}")
            raise

        token_usage = usage_callback.usage_metadata
        logger.info(f"OCR 성공 - 이미지 크기: {image_size_kb:.2f}KB. Token Usage: {token_usage}")

        return response.content
