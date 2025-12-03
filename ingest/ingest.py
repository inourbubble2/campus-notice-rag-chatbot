import logging
from dotenv import load_dotenv
from typing import List

from sqlalchemy import RowMapping

from .chunk_embed import build_documents_from_parsed
from services.database_service import (
    fetch_parsed_records_by_ids,
    fetch_parsed_records_by_date_range
)
from services.embed_service import embed_and_store_documents



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

async def _ingest_parsed_rows(rows: List[RowMapping]) -> dict:
    logger.info(f"Processing {len(rows)} parsed announcements for embedding...")

    docs = build_documents_from_parsed(rows)
    logger.info(f"Generated {len(docs)} document chunks from {len(rows)} announcements")

    await embed_and_store_documents(
        docs=docs,
    )

    logger.info(f"✓ Successfully embedded and stored {len(docs)} chunks")

    return {
        "success": True,
        "announcement_count": len(rows),
        "chunk_count": len(docs)
    }


async def ingest_by_ids(ids: List[int] = None) -> dict:
    rows = fetch_parsed_records_by_ids(ids)
    return await _ingest_parsed_rows(rows)


async def ingest_by_date_range(from_date: str, to_date: str) -> dict:
    rows = fetch_parsed_records_by_date_range(from_date, to_date)
    return await _ingest_parsed_rows(rows)
