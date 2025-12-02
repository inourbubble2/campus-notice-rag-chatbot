from app.settings import get_settings
from services.ocr.base import BaseOCRService
from services.ocr.gemini_ocr_service import GeminiOCRService
from services.ocr.upstage_ocr_service import UpstageOCRService


def get_ocr_service() -> BaseOCRService:
    """
    Factory function to get the configured OCR service.

    Returns:
        An instance of a class inheriting from BaseOCRService.
    """
    ocr_provider = get_settings().ocr_provider.lower()

    if ocr_provider == "gemini":
        return GeminiOCRService()
    elif ocr_provider == "upstage":
        return UpstageOCRService()

    return GeminiOCRService()
