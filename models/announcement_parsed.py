# models/announcement_parsed.py
"""중간 테이블(announcement_parsed) 관련 모델"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class AnnouncementParsed(BaseModel):
    """중간 테이블 레코드 (announcement_parsed)"""
    id: Optional[int] = None
    announcement_id: int
    title: str
    written_at: Optional[datetime] = None

    # 정제된 텍스트
    cleaned_text: Optional[str] = None
    ocr_text: Optional[str] = None

    # 구조화된 데이터
    application_period_start: Optional[datetime] = None
    application_period_end: Optional[datetime] = None
    target_departments: Optional[List[str]] = None
    target_grades: Optional[List[int]] = None
    tags: Optional[List[str]] = None
    structured_info: Optional[Dict[str, Any]] = None

    # 처리 상태
    error_message: Optional[str] = None

    # 타임스탬프
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AdditionalInfoItem(BaseModel):
    key: str = Field(..., description="추가 정보의 키")
    value: str = Field(..., description="추가 정보의 값")

class AnnouncementParsedInfo(BaseModel):
    """LLM으로 추출한 구조화된 정보"""
    application_period_start: Optional[str] = None  # "YYYY-MM-DD" or "YYYY-MM-DD HH:MM"
    application_period_end: Optional[str] = None
    target_departments: Optional[List[str]] = None
    target_grades: Optional[List[int]] = None
    tags: Optional[List[str]] = None
    additional_info: Optional[List[AdditionalInfoItem]] = Field(
        None, description="자유 형식의 추가 정보 (key-value 쌍 리스트)"
    )
