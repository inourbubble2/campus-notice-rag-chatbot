import logging
from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""],
)


def _build_enhanced_text(row: Dict) -> str:
    """
    announcement_parsed의 row로부터 텍스트 증강
    """
    parts = []
    # 1. 본문 (cleaned_text)
    if row.get('cleaned_text'):
        parts.append(row['cleaned_text'])

    # 2. OCR 텍스트
    if row.get('ocr_text'):
        parts.append(row['ocr_text'])

    return "\n".join(parts)


def build_documents_from_parsed(parsed_rows: List[Dict]) -> List[Document]:
    docs = []

    for row in parsed_rows:
        try:
            # 텍스트 증강
            enhanced_text = _build_enhanced_text(row)

            # 청킹
            text_chunks = splitter.split_text(enhanced_text)

            for i, chunk_text in enumerate(text_chunks):
                # 메타데이터 구성
                page_content = row["title"] + "\n" + chunk_text
                metadata = {
                    "announcement_id": row["announcement_id"],
                    "chunk_index": i,
                    "title": row["title"],
                    "written_at": row["written_at"].isoformat() if row.get("written_at") else None,

                    # 원본 공지사항 메타데이터 (JOIN으로 가져옴)
                    "board": row.get("board"),
                    "author": row.get("author"),
                    "major": row.get("major"),
                    "url": row.get("url"),
                }
                docs.append(Document(page_content=page_content, metadata=metadata))

        except Exception as e:
            logging.error(f"Error processing announcement {row.get('announcement_id', 'unknown')}: {e}")
            continue

    return docs
