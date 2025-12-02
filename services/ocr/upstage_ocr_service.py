import time
import base64
import asyncio
import logging
import tempfile
import os
from tenacity import (
  retry,
  stop_after_attempt,
  wait_exponential,
  retry_if_exception_type,
  before_sleep_log
)
from langchain_upstage import UpstageDocumentParseLoader
from services.ocr.base import BaseOCRService

logger = logging.getLogger(__name__)


class UpstageOCRService(BaseOCRService):
    def _extract_text_from_image_sync(self, img_base64: str) -> tuple[str, float, float]:
        """
        Upstage DocumentParseLoader를 사용하여 이미지에서 텍스트 추출 (동기 함수).

        Args:
            img_base64: Base64로 인코딩된 이미지 데이터

        Returns:
            tuple[str, float, float]: (OCR 결과 텍스트, 이미지 크기(KB), 소요 시간(ms))
        """
        # Base64 디코딩
        img_bytes = base64.b64decode(img_base64)
        image_size_kb = len(img_bytes) / 1024

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(img_bytes)
            tmp_file_path = tmp_file.name

        try:
            logger.info(f"Upstage OCR 요청 시작 - 이미지 크기: {image_size_kb:.2f}KB")
            start_time = time.time()

            # UpstageDocumentParseLoader 사용 (ocr="force"로 강제 OCR 수행)
            loader = UpstageDocumentParseLoader(tmp_file_path, ocr="force")
            pages = loader.load()

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # 모든 페이지의 텍스트를 합침
            result_text = '\n'.join([page.page_content for page in pages])

            logger.info(f"OCR 성공 - 이미지 크기: {image_size_kb:.2f}KB, 소요 시간: {duration_ms:.2f}ms, 결과 길이: {len(result_text)}")
            return result_text, image_size_kb, duration_ms

        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    @retry(
        stop=stop_after_attempt(3),  # 최대 3회 시도
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 2초, 4초, 8초 대기
        retry=retry_if_exception_type((TimeoutError, asyncio.TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def extract_text_from_image(self, img_base64: str) -> str:
        """
        Upstage DocumentParseLoader를 사용하여 이미지에서 텍스트 추출 (비동기 래퍼).

        Args:
            img_base64: Base64로 인코딩된 이미지 데이터

        Returns:
            OCR 결과 텍스트
        """
        try:
            loop = asyncio.get_event_loop()
            result_text, image_size_kb, duration_ms = await loop.run_in_executor(
                None,
                self._extract_text_from_image_sync,
                img_base64
            )

            return result_text

        except Exception as e:
            logger.error(f"OCR 요청 실패 - 에러: {type(e).__name__}: {str(e)}")
            raise
