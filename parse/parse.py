# ingest/parse.py
"""
공지사항 구조화 처리 모듈
원본 → 중간테이블: HTML 정제, OCR
"""
import logging
import asyncio
from typing import List

from sqlalchemy import RowMapping

from services.html_processing_service import get_plain_text, extract_image_urls, html_to_text
from services.database_service import (
    fetch_rows_by_ids,
    fetch_rows_by_date_range,
    upsert_processed_record,
)
from models.announcement_parsed import AnnouncementParsed
from services.ocr.base import BaseOCRService

logger = logging.getLogger(__name__)

# 동시 처리 수 제한
MAX_CONCURRENT_TASKS = 20
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)


async def _process_single_announcement(row: RowMapping, ocr_service: BaseOCRService) -> AnnouncementParsed:
    """Semaphore로 동시 처리 수를 제한하면서 단일 공지사항 처리"""
    async with _semaphore:
        announcement_id = row["id"]
        title = row["title"]
        written_at = row["written_at"]
        html = row["html"]

        try:
            logger.info(f"Processing announcement {announcement_id}: {title}")

            cleaned_text = get_plain_text(html)

            try:
                # 1. 이미지 URL 추출
                image_urls = extract_image_urls(html)

                # 2. OCR 서비스에 위임 (병렬 처리 및 에러 핸들링 포함)
                ocr_text, ocr_error = await ocr_service.extract_text_from_urls(image_urls)

                if ocr_error:
                    logger.warning(f"Announcement {announcement_id}: {ocr_error}")

            except Exception as ocr_exc:
                ocr_error = f"OCR orchestration failed: {type(ocr_exc).__name__}: {str(ocr_exc)}"
                logger.error(f"OCR completely failed for announcement {announcement_id}: {ocr_error}")
                ocr_text = None

            processed_data = AnnouncementParsed(
                announcement_id=announcement_id,
                title=title,
                written_at=written_at,
                cleaned_text=cleaned_text,
                ocr_text=ocr_text,
                error_message=ocr_error,  # OCR 실패 시 에러 메시지 기록
            )

            upsert_processed_record(processed_data)
            logger.info(f"✓ Successfully processed announcement {announcement_id}")
            return processed_data

        except Exception as e:
            logger.error(f"✗ Failed to process announcement {announcement_id}: {e}")
            failed_data = AnnouncementParsed(
                announcement_id=announcement_id,
                title=title,
                written_at=written_at,
                error_message=str(e),
            )
            upsert_processed_record(failed_data)
            return failed_data


async def process_announcements_by_ids(ids: List[int], ocr_service: BaseOCRService = None) -> List[AnnouncementParsed]:
    rows = fetch_rows_by_ids(ids)
    tasks = [_process_single_announcement(row, ocr_service) for row in rows]
    results = await asyncio.gather(*tasks)
    return results


async def process_announcements_by_date_range(from_date: str, to_date: str, ocr_service: BaseOCRService = None) -> List[AnnouncementParsed]:
    rows = fetch_rows_by_date_range(from_date, to_date)
    tasks = [_process_single_announcement(row, ocr_service) for row in rows]
    results = await asyncio.gather(*tasks)
    return results
