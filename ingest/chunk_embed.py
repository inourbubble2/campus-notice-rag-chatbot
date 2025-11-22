import logging
from typing import List, Dict

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""],
)


def _build_enhanced_text(row: Dict) -> str:
    """
    announcement_parsed의 row로부터 텍스트 증강 (구조화 정보 포함)
    """
    parts = []
    # 1. 본문 (cleaned_text)
    if row.get('cleaned_text'):
        parts.append(row['cleaned_text'])

    # 2. OCR 텍스트
    if row.get('ocr_text'):
        parts.append(row['ocr_text'])

    # 3. 구조화 정보를 자연어로 추가
    if row.get('tags') and len(row['tags']) > 0:
        parts.append(f"관련 태그: {', '.join(row['tags'])}")

    if row.get('target_departments') and len(row['target_departments']) > 0:
        parts.append(f"대상 학과: {', '.join(row['target_departments'])}")

    if row.get('target_grades') and len(row['target_grades']) > 0:
        grades = ', '.join([f"{g}학년" for g in row['target_grades']])
        parts.append(f"대상 학년: {grades}")

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

                    # 필터링용 구조화 데이터
                    "tags": row.get("tags") or [],
                    "target_departments": row.get("target_departments") or [],
                    "target_grades": row.get("target_grades") or [],
                    "application_period_start": row["application_period_start"].isoformat() if row.get("application_period_start") else None,
                    "application_period_end": row["application_period_end"].isoformat() if row.get("application_period_end") else None,
                }

                docs.append(Document(page_content=page_content, metadata=metadata))

        except Exception as e:
            logging.error(f"Error processing announcement {row.get('announcement_id', 'unknown')}: {e}")
            continue

    return docs
