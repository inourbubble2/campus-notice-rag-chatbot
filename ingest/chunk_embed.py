import asyncio
import logging
from itertools import chain
from typing import List, Dict

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .preprocess import get_text, get_text_with_ocr

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=160,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""],
)

async def _process_row(row: Dict) -> List[Document]:
  try:
    body_text = await get_text_with_ocr(row["html"])
    if not body_text:
      # fallback: 텍스트만이라도
      body_text = get_text(row["html"])

    title = row["title"]
    full_text = title + ' ' + body_text
    row_docs = []
    for i, chunk in enumerate(splitter.split_text(full_text)):
      metadata = {
        "title": row["title"],
        "board": row["board"],
        "author": row["author"],
        "major": row["major"],
        "written_at": (row["written_at"].isoformat() if row["written_at"] else None),
        "created_at": row["created_at"].isoformat(),
        "modified_at": row["modified_at"].isoformat(),
        "url": row["url"],
        "chunk_index": i,
        "announcement_id": int(row["id"]),
      }
      row_docs.append(Document(page_content=chunk, metadata=metadata))
    return row_docs
  except Exception as e:
    logging.error(f"Error processing announcement {row.get('id', 'unknown')}: {e}")
    return []

async def build_documents(rows: List[Dict]) -> List[Document]:
  tasks = [_process_row(row) for row in rows]
  results = await asyncio.gather(*tasks)
  return list(chain.from_iterable(results))
