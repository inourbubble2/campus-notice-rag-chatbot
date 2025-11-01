# services/retriever_service.py
"""
벡터 스토어 검색 서비스.
Retriever를 사용하여 관련 문서를 검색합니다.
"""
from typing import List
from langchain_core.documents import Document

from app.deps import get_settings, get_retriever


async def retriever_search(query: str, k: int, fetch_k: int = 40) -> List[Document]:
    """
    동적 k/fetch_k를 지원하는 검색 함수 (MMR 기본 활성화).

    Args:
        query: 검색 쿼리
        k: 최종 반환할 문서 개수
        fetch_k: MMR 전 후보 풀 크기

    Returns:
        검색된 Document 리스트
    """
    cfg = get_settings()
    retriever = get_retriever(
        k=k,
        fetch_k=fetch_k,
        mmr=cfg.retriever_mmr
    )
    return await retriever.ainvoke(query)


if __name__ == "__main__":
    import asyncio

    async def main():
        # test
        test_query = "랜선 야학"
        print(f"Query: {test_query}\n")

        results = await retriever_search(query=test_query, k=5, fetch_k=20)

        print(f"Found {len(results)} documents:\n")
        for i, doc in enumerate(results, 1):
            print(f"=== Document {i} ===")
            # print(f"Content: {doc.page_content[:100]}...")
            print(f"Metadata: {doc.metadata}")
            print()

    asyncio.run(main())
