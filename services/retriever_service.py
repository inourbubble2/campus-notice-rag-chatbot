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


async def retriever_search(
    query: str,
    k: int,
    fetch_k: int = 40,
) -> List[Document]:
    cfg = get_settings()
    vectorstore = get_vectorstore()

    vectorstore = get_vectorstore()

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
        docs_with_score = await vectorstore.asimilarity_search_with_score(query, k=search_k)
        docs = []
        for doc, score in docs_with_score:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["score"] = score
            docs.append(doc)

    logger.info(f"Vector search returned {len(docs)} documents")

    return docs


def rerank_documents(query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
    """
    CrossEncoder(Dongjin-kr/ko-reranker)를 사용하여 문서 리랭킹 수행.
    """
    if not docs:
        return []

    try:
        from app.deps import get_reranker
        
        ranker = get_reranker()
        
        # (query, passage) 쌍 생성
        pairs = [[query, doc.page_content] for doc in docs]
        
        # 점수 계산
        scores = ranker.predict(pairs)
        
        # 점수와 함께 문서 저장
        scored_docs = []
        for doc, score in zip(docs, scores):
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["score"] = float(score)
            scored_docs.append(doc)
            
        # 점수 내림차순 정렬
        scored_docs.sort(key=lambda x: x.metadata["score"], reverse=True)
        
        # 상위 N개 반환
        reranked_docs = scored_docs[:top_n]
            
        logger.info(f"Reranked {len(docs)} -> {len(reranked_docs)} documents (Top score: {reranked_docs[0].metadata['score']:.4f})")
        return reranked_docs
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return docs[:top_n]
