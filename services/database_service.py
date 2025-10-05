# services/database_service.py
from typing import List
from sqlalchemy import text, RowMapping
from app.deps import get_engine


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
