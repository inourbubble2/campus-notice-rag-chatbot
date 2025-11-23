import logging
from services.retriever_service import retriever_search, rerank_documents
from chat.schema import RAGState

logger = logging.getLogger(__name__)

BASE_K = 6
K_STEP = 4
K_MAX = 20  # 리랭킹을 위해 넉넉하게 가져옴

async def retrieve_node(state: RAGState) -> RAGState:
    rw = state.get("rewrite") or {}
    q = rw.get("query") or state["question"]
    keywords = rw.get("keywords", [])
    
    # 키워드를 쿼리 뒤에 붙여서 검색 강화
    if keywords:
        q = f"{q} {' '.join(keywords)}"

    # 1. Vector Search
    logger.info(f"Retrieving documents: k={K_MAX}, attempt={state.get('attempt', 0)}")
    docs = await retriever_search(q, k=K_MAX)

    # 2. Reranking
    logger.info("Reranking documents...")
    reranked_docs = rerank_documents(q, docs, top_n=BASE_K)

    logger.info(f"Retrieved {len(reranked_docs)} documents after reranking")
    state["docs"] = reranked_docs
    return state
