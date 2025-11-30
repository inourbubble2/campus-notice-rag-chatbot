import logging
import asyncio
from abc import ABC, abstractmethod
from services.image_download_service import download_image_as_base64

logger = logging.getLogger(__name__)

class BaseOCRService(ABC):
    @abstractmethod
    async def extract_text_from_image(self, img_base64: str) -> str:
        """
        Extract text from a base64 encoded image.
        
        Args:
            img_base64: Base64 encoded image string.
            
        Returns:
            Extracted text.
        """
        pass

    async def extract_text_from_url(self, url: str) -> str:
        """
        URL에서 이미지를 다운로드하고 OCR 수행.
        
        Args:
            url: 이미지 URL
            
        Returns:
            Extracted text.
        """
        img_base64 = await download_image_as_base64(url)
        return await self.extract_text_from_image(img_base64)

    async def extract_text_from_urls(self, urls: list[str]) -> tuple[str | None, str | None]:
        """
        여러 이미지 URL에 대해 병렬로 OCR을 수행하고 결과를 반환.
        
        Args:
            urls: 이미지 URL 리스트
            
        Returns:
            tuple[str | None, str | None]: (OCR 텍스트, 에러 요약)
                - OCR 텍스트: 성공한 이미지들의 OCR 결과 (없으면 None)
                - 에러 요약: 실패한 이미지가 있을 경우 에러 메시지 (없으면 None)
        """
        if not urls:
            return None, None

        # 병렬 OCR 수행
        ocr_tasks = [self.extract_text_from_url(url) for url in urls]
        ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)

        # 결과 처리
        valid_ocr_results = []
        failed_images = []

        for i, result in enumerate(ocr_results):
            if isinstance(result, str):
                if result.strip():
                    valid_ocr_results.append(result)
            elif isinstance(result, Exception):
                error_msg = f"Image {i+1}: {type(result).__name__}: {str(result)}"
                failed_images.append(error_msg)
                logger.error(
                    f"OCR failed for image {i+1}: {str(result)}",
                    exc_info=result
                )

        # 에러 요약 생성
        ocr_error = None
        if failed_images:
            if not valid_ocr_results:
                ocr_error = f"All {len(urls)} images failed OCR: {'; '.join(failed_images[:3])}"
            else:
                ocr_error = f"Partial OCR failure: {len(failed_images)}/{len(urls)} images failed"
        
        ocr_text = "\n".join(valid_ocr_results) if valid_ocr_results else None
        
        return ocr_text, ocr_error
