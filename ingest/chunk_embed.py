import asyncio
import logging
from itertools import chain
from typing import List, Dict

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from services.html_processing_service import get_plain_text, get_ocr_text

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=160,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""],
)

async def _process_row(row: Dict) -> List[Document]:
  try:
    cleaned_text = get_plain_text(row["html"])
    ocr_text = await get_ocr_text(row["html"])
    body_text = "\n".join([cleaned_text, ocr_text])

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
