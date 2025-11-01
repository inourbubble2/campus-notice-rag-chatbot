# services/database_service.py
from typing import List, Optional
from sqlalchemy import text, RowMapping
from app.deps import get_engine
from models.announcement_parsed import AnnouncementParsed
import json


# ========== 원본 공지사항 조회 ==========

def fetch_rows_by_ids(ids: List[int]) -> List[RowMapping]:
    """ID 목록으로 공지사항 조회."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
         SELECT a.id, a.title, a.board, a.author, a.major, a.written_at, a.created_at,
                a.modified_at, a.target_url AS url, ad.html
         FROM public.announcement a
                  JOIN public.announcement_detail ad ON ad.id = a.announcementdetail_id
         WHERE a.id = ANY(:ids)
         ORDER BY a.id
         """), {"ids": ids}).mappings().all()
        return list(rows)


def fetch_rows_by_date_range(from_date: str, to_date: str) -> List[RowMapping]:
    """날짜 범위로 공지사항 조회."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
         SELECT a.id, a.title, a.board, a.author, a.major, a.written_at, a.created_at,
                a.modified_at, a.target_url AS url, ad.html
         FROM public.announcement a
                  JOIN public.announcement_detail ad ON ad.id = a.announcementdetail_id
         WHERE a.written_at >= :from_date AND a.written_at <= :to_date
         ORDER BY a.written_at DESC
         """), {"from_date": from_date, "to_date": to_date}).mappings().all()
        return list(rows)


# ========== 중간 테이블 (announcement_parsed) CRUD ==========

def upsert_processed_record(data: AnnouncementParsed) -> int:
    """
    중간 테이블에 레코드 삽입/업데이트 (UPSERT).
    announcement_id가 이미 존재하면 업데이트, 없으면 삽입.
    반환: processed record ID
    """
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO public.announcement_parsed (
                announcement_id, title, written_at, cleaned_text, ocr_text,
                application_period_start, application_period_end,
                target_departments, target_grades, tags, structured_info,
                error_message
            ) VALUES (
                :announcement_id, :title, :written_at, :cleaned_text, :ocr_text,
                :application_period_start, :application_period_end,
                :target_departments, :target_grades, :tags, :structured_info,
                :error_message
            )
            ON CONFLICT (announcement_id) DO UPDATE SET
                title = EXCLUDED.title,
                written_at = EXCLUDED.written_at,
                cleaned_text = EXCLUDED.cleaned_text,
                ocr_text = EXCLUDED.ocr_text,
                application_period_start = EXCLUDED.application_period_start,
                application_period_end = EXCLUDED.application_period_end,
                target_departments = EXCLUDED.target_departments,
                target_grades = EXCLUDED.target_grades,
                tags = EXCLUDED.tags,
                structured_info = EXCLUDED.structured_info,
                error_message = EXCLUDED.error_message,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """), {
            "announcement_id": data.announcement_id,
            "title": data.title,
            "written_at": data.written_at,
            "cleaned_text": data.cleaned_text,
            "ocr_text": data.ocr_text,
            "application_period_start": data.application_period_start,
            "application_period_end": data.application_period_end,
            "target_departments": data.target_departments,
            "target_grades": data.target_grades,
            "tags": data.tags,
            "structured_info": json.dumps(data.structured_info) if data.structured_info else None,
            "error_message": data.error_message,
        })
        conn.commit()
        return result.scalar_one()


def fetch_processed_records_by_ids(ids: List[int]) -> List[RowMapping]:
    """ID 목록으로 중간 테이블 레코드들 조회 (announcement_id 기준)."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT * FROM public.announcement_parsed
            WHERE announcement_id = ANY(:ids)
            ORDER BY announcement_id
        """), {"ids": ids}).mappings().all()
        return list(rows)
