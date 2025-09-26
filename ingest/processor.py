import os
import logging
import time
from dotenv import load_dotenv
from typing import List

from sqlalchemy import RowMapping

from .chunk_embed import build_documents
from app.deps import get_vectorstore
from app.database import fetch_rows_by_ids, fetch_rows_by_date_range

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "uos_announcement")


def _ingest_rows(rows: List[RowMapping], description: str = ""):
    """공통 ingest 로직을 처리하는 private 함수."""
    docs = build_documents(rows)

    if description:
        logger.info(f"Built {len(docs)} document chunks {description}")
    else:
        logger.info(f"Built {len(docs)} document chunks from {len(rows)} announcements")

    if not docs:
        logger.warning("No documents to process")
        return

    vector_store = get_vectorstore()
    total_tokens = sum(len(doc.page_content.split()) for doc in docs)
    logger.info(f"Starting embedding for ~{total_tokens} tokens across {len(docs)} chunks")

    start_time = time.time()
    vector_store.add_documents(documents=docs)
    end_time = time.time()

    processing_time = end_time - start_time
    logger.info(f"Embedding completed in {processing_time:.2f} seconds")
    logger.info(f"Processed ~{total_tokens} tokens at {total_tokens/processing_time:.1f} tokens/second")
    logger.info(f"Indexed {len(docs)} chunks into collection '{COLLECTION_NAME}'")


def ingest_by_ids(ids: List[int] = None):
    """ID 목록으로 공지사항을 ingest합니다."""
    rows = fetch_rows_by_ids(ids)
    _ingest_rows(rows, f"from {len(ids)} announcement IDs")


def ingest_by_date_range(from_date: str, to_date: str):
    """날짜 범위로 공지사항을 ingest합니다."""
    rows = fetch_rows_by_date_range(from_date, to_date)
    _ingest_rows(rows, f"from {len(rows)} announcements between {from_date} and {to_date}")
