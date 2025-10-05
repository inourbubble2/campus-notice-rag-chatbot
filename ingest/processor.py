import logging
from dotenv import load_dotenv
from typing import List

from sqlalchemy import RowMapping

from .chunk_embed import build_documents
from app.deps import get_settings
from services.database_service import fetch_rows_by_ids, fetch_rows_by_date_range
from services.embed_service import embed_and_store_documents
from models.metrics import get_ocr_metrics_aggregator, IngestResult


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

async def _ingest_rows(rows: List[RowMapping]) -> IngestResult:
    """공통 ingest 로직을 처리."""

    # OCR 메트릭 집계기 초기화
    aggregator = get_ocr_metrics_aggregator()
    aggregator.reset()

    # 문서 빌드
    docs = await build_documents(rows)

    # 임베딩 생성 및 저장
    embedding_metrics = await embed_and_store_documents(
        docs=docs,
        announcement_count=len(rows)
    )

    return IngestResult(embedding_metrics=embedding_metrics, ocr_metrics=aggregator.get_summary())


async def ingest_by_ids(ids: List[int] = None) -> IngestResult:
    rows = fetch_rows_by_ids(ids)
    return await _ingest_rows(rows)


async def ingest_by_date_range(from_date: str, to_date: str) -> IngestResult:
    rows = fetch_rows_by_date_range(from_date, to_date)
    return await _ingest_rows(rows)
