# services/retriever_service.py
"""
벡터 스토어 검색 서비스.
MMR 및 메타데이터 필터링을 지원하는 검색 기능을 제공합니다.
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from app.deps import get_settings, get_vectorstore

logger = logging.getLogger(__name__)


def _check_array_overlap(doc_array: List[Any], filter_values: List[Any]) -> bool:
    """
    배열 overlap 체크: 필터 값 중 하나라도 문서 배열에 포함되어 있으면 True

    Args:
        doc_array: 문서의 메타데이터 배열 (예: ["조경학과", "건축학과"])
        filter_values: 필터링할 값들 (예: ["조경학과"])

    Returns:
        overlap 여부
    """
    if not doc_array or not filter_values:
        return False

    # 배열이 None이거나 빈 배열이면 False
    doc_set = set(doc_array) if isinstance(doc_array, list) else set()
    filter_set = set(filter_values)

    return bool(doc_set & filter_set)  # 교집합이 있으면 True


async def retriever_search(
    query: str,
    k: int,
    fetch_k: int = 40,
    filters: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    동적 k/fetch_k를 지원하는 검색 함수 with MMR 및 후처리 필터링.

    MMR (Maximal Marginal Relevance):
    - 유사도와 다양성을 동시에 고려
    - 중복 문서 최소화

    후처리 필터링:
    - LangChain PGVector는 배열 필터를 지원하지 않으므로
    - Python에서 정확하게 필터링

    Args:
        query: 검색 쿼리
        k: 최종 반환할 문서 개수
        fetch_k: MMR 전 후보 풀 크기
        filters: 메타데이터 필터 (departments, grades, tags)

    Returns:
        검색된 Document 리스트
    """
    cfg = get_settings()
    vectorstore = get_vectorstore()

    # 필터가 있으면 더 많은 문서를 가져와서 후처리
    has_filters = filters and any([
        filters.get("departments"),
        filters.get("grades"),
        filters.get("tags")
    ])

    if has_filters:
        # 필터링 시 손실을 고려해서 3배 더 가져옴
        search_k = k * 3
        logger.info(f"Fetching {search_k} documents for post-filtering (target: {k})")
    else:
        search_k = k

    # MMR 검색
    if cfg.retriever_mmr:
        # MMR: 유사도 + 다양성 고려
        lambda_mult = cfg.retriever_lambda_mult
        effective_fetch_k = max(fetch_k, search_k * 2)  # fetch_k는 최소 search_k의 2배
        logger.info(f"Using MMR search (k={search_k}, fetch_k={effective_fetch_k}, lambda={lambda_mult})")
        docs = await vectorstore.amax_marginal_relevance_search(
            query,
            k=search_k,
            fetch_k=effective_fetch_k,
            lambda_mult=lambda_mult
        )
    else:
        # 일반 유사도 검색
        logger.info(f"Using similarity search (k={search_k})")
        docs = await vectorstore.asimilarity_search(query, k=search_k)

    logger.info(f"Vector search returned {len(docs)} documents")

    # 후처리 필터링
    if has_filters:
        filtered_docs = []

        for doc in docs:
            md = doc.metadata or {}
            matched = False

            # departments 필터: target_departments 배열과 overlap 확인
            if filters.get("departments"):
                if _check_array_overlap(
                    md.get("target_departments", []),
                    filters["departments"]
                ):
                    matched = True

            # grades 필터: target_grades 배열과 overlap 확인
            if not matched and filters.get("grades"):
                if _check_array_overlap(
                    md.get("target_grades", []),
                    filters["grades"]
                ):
                    matched = True

            # tags 필터: tags 배열과 overlap 확인
            if not matched and filters.get("tags"):
                if _check_array_overlap(
                    md.get("tags", []),
                    filters["tags"]
                ):
                    matched = True

            if matched:
                filtered_docs.append(doc)

        logger.info(f"Post-filtering: {len(docs)} → {len(filtered_docs)} documents")

        # 필터링 결과가 충분하지 않으면 필터 없는 결과로 fallback
        if len(filtered_docs) < k // 2:
            logger.warning(f"Filtered results ({len(filtered_docs)}) below threshold ({k // 2}). Using unfiltered results.")
            docs = docs[:k]
        else:
            # 필터링 후 k개만 반환
            docs = filtered_docs[:k]

    return docs


if __name__ == "__main__":
    import asyncio

    async def main():
        # test
        test_query = "조경학과"
        print(f"Query: {test_query}\n")

        results = await retriever_search(query=test_query, k=5, fetch_k=20)

        print(f"Found {len(results)} documents:\n")
        for i, doc in enumerate(results, 1):
            print(f"=== Document {i} ===")
            print(f"Content: {doc.page_content}...")
            print(f"Metadata: {doc.metadata}")
            print()

    asyncio.run(main())
