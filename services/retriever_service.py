# services/retriever_service.py
"""
벡터 스토어 검색 서비스.
"""
import logging
from typing import List
from langchain_core.documents import Document

from app.deps import get_vectorstore
from services.database_service import fetch_parsed_records_by_ids

logger = logging.getLogger(__name__)
async def retriever_search(query: str, k: int) -> List[Document]:
    vectorstore = get_vectorstore()

    child_docs = await vectorstore.asimilarity_search_with_score(query, k=k)

    parent_ids = []
    id_to_score = {}

    for doc, score in child_docs:
        pid = doc.metadata.get("announcement_id")
        if pid and pid not in id_to_score:
            id_to_score[pid] = score
            parent_ids.append(pid)

        if len(parent_ids) >= k:
            break

    if not parent_ids:
        return []

    rows = fetch_parsed_records_by_ids(parent_ids)
    row_map = {row['announcement_id']: row for row in rows}

    final_docs = []
    for pid in parent_ids:
        row = row_map.get(pid)

        parts = [f"제목: {row['title']}"]
        if row.get('cleaned_text'):
            parts.append(row['cleaned_text'])
        if row.get('ocr_text'):
            parts.append(f"\n[이미지/첨부파일 내용]\n{row['ocr_text']}")

        full_text = "\n\n".join(parts)

        # 메타데이터
        metadata = {
            "announcement_id": row['announcement_id'],
            "title": row['title'],
            "board": row.get('board'),
            "author": row.get('author'),
            "major": row.get('major'),
            "url": row.get('url'),
            "written_at": row['written_at'].isoformat() if row.get('written_at') else None,
            "score": id_to_score[pid]
        }

        final_docs.append(Document(page_content=full_text, metadata=metadata))

    return final_docs
