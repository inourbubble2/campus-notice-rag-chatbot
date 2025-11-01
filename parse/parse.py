# ingest/parse.py
"""
공지사항 구조화 처리 모듈
원본 → 중간테이블: HTML 정제, OCR, 구조화된 정보 추출
"""
import logging
import asyncio
from typing import List
from datetime import datetime

from sqlalchemy import RowMapping

from services.html_processing_service import get_plain_text, get_ocr_text
from services.database_service import (
    fetch_rows_by_ids,
    fetch_rows_by_date_range,
    upsert_processed_record,
)
from services.extraction_service import extract_structured_info
from models.announcement_parsed import AnnouncementParsed

logger = logging.getLogger(__name__)


def _parse_datetime(date_str: str) -> datetime:
    """날짜 문자열을 datetime으로 파싱. 실패 시 None 반환."""
    try:
        return datetime.fromisoformat(date_str.replace(" ", "T"))
    except:
        return None


async def _process_single_announcement(row: RowMapping) -> AnnouncementParsed:
    announcement_id = row["id"]
    title = row["title"]
    written_at = row["written_at"]
    html = row["html"]

    try:
        logger.info(f"Processing announcement {announcement_id}: {title}")

        # 1. HTML 정제 (텍스트 추출)
        cleaned_text = get_plain_text(html)

        # 2. OCR 처리
        ocr_text = await get_ocr_text(html)

        # 3. 전체 텍스트 결합 (정제된 텍스트 + OCR)
        full_text = f"{cleaned_text}\n{ocr_text}" if ocr_text else cleaned_text

        # 4. LLM으로 구조화된 정보 추출
        structured_info = extract_structured_info(full_text, title)

        # 5. 중간 테이블 데이터 준비
        processed_data = AnnouncementParsed(
            announcement_id=announcement_id,
            title=title,
            written_at=written_at,
            cleaned_text=cleaned_text,
            ocr_text=ocr_text if ocr_text else None,
        )

        # 6. 구조화된 정보 추가 (추출 성공 시)
        if structured_info:
            processed_data.application_period_start = _parse_datetime(structured_info.application_period_start) if structured_info.application_period_start else None
            processed_data.application_period_end = _parse_datetime(structured_info.application_period_end) if structured_info.application_period_end else None
            processed_data.target_departments = structured_info.target_departments
            processed_data.target_grades = structured_info.target_grades
            processed_data.tags = structured_info.tags
            if structured_info.additional_info:
                processed_data.structured_info = [
                    i.model_dump() if hasattr(i, "model_dump") else dict(i)
                    for i in structured_info.additional_info
                ]
            else:
                processed_data.structured_info = None

        # 7. 중간 테이블에 UPSERT
        upsert_processed_record(processed_data)
        logger.info(f"✓ Successfully processed announcement {announcement_id}")
        return processed_data

    except Exception as e:
        logger.error(f"✗ Failed to process announcement {announcement_id}: {e}")
        # 실패 상태 저장
        failed_data = AnnouncementParsed(
            announcement_id=announcement_id,
            title=title,
            written_at=written_at,
            error_message=str(e),
        )
        upsert_processed_record(failed_data)
        return failed_data


async def process_announcements_by_ids(ids: List[int]) -> List[AnnouncementParsed]:
    # 원본 테이블에서 조회
    rows = fetch_rows_by_ids(ids)

    # 병렬 처리
    tasks = [_process_single_announcement(row) for row in rows]
    results = await asyncio.gather(*tasks)
    return results


async def process_announcements_by_date_range(from_date: str, to_date: str) -> List[AnnouncementParsed]:
    # 원본 테이블에서 조회
    rows = fetch_rows_by_date_range(from_date, to_date)

    # 병렬 처리
    tasks = [_process_single_announcement(row) for row in rows]
    results = await asyncio.gather(*tasks)
    return results
