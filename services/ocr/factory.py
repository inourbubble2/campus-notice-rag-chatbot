import os
from services.ocr.base import BaseOCRService
from services.ocr.upstage_ocr_service import UpstageOCRService
from services.ocr.gemini_ocr_service import GeminiOCRService

def get_ocr_service() -> BaseOCRService:
    """
    Factory function to get the configured OCR service.
    
    Returns:
        An instance of a class inheriting from BaseOCRService.
    """
    ocr_provider = os.getenv("OCR_PROVIDER", "gemini").lower()
    
    if ocr_provider == "gemini":
        return GeminiOCRService()
    elif ocr_provider == "upstage":
        return UpstageOCRService()
    
    return GeminiOCRService()
