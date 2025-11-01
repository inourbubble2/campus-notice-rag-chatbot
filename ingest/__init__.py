# ingest package
"""
데이터 인제스트 패키지

- processor: 데이터 처리 및 임베딩 메인 로직
- chunk_embed: 문서 청킹 및 임베딩 준비
"""

from .processor import ingest_by_ids, ingest_by_date_range

__all__ = [
    "ingest_by_ids",
    "ingest_by_date_range",
]