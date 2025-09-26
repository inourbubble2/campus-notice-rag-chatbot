import logging
from typing import List, Dict

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .preprocess import normalize_text, compose_text_with_ocr

def build_documents(rows: List[Dict]) -> List[Document]:
  docs: List[Document] = []
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=800,
      chunk_overlap=160,
      separators=["\n\n", "\n", ".", "!", "?", " ", ""],
  )

  for row in rows:
    body_text = compose_text_with_ocr(row["html"])
    if not body_text:
      # fallback: 텍스트만이라도
      body_text = normalize_text(row["html"])

    title = row["title"]
    full_text = f"[{title} - {body_text}"
    logging.debug(full_text[:100], '...')

    for i, chunk in enumerate(splitter.split_text(full_text)):
      meta = {
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
      docs.append(Document(page_content=chunk, metadata=meta))
  return docs
