import logging
from services.retriever_service import retriever_search
from chat.schema import RAGState

logger = logging.getLogger(__name__)

BASE_K = 6
K_STEP = 4
K_MAX = 20

async def retrieve_node(state: RAGState) -> RAGState:
    rw = state.get("rewrite") or {}
    q = rw.get("query") or state["question"]
    filters = rw.get("filters", {})

    # 시도 횟수에 따라 k 증가
    k = min(BASE_K + state.get("attempt", 0) * K_STEP, K_MAX)

    # 필터가 있는지 확인
    has_filters = any([
        filters.get("departments"),
        filters.get("grades"),
        filters.get("tags")
    ])

    if has_filters:
        logger.info(f"Retrieving documents with filters: k={k}, attempt={state.get('attempt', 0)}")
        docs = await retriever_search(q, k=k, filters=filters)
    else:
        logger.info(f"Retrieving documents: k={k}, attempt={state.get('attempt', 0)}")
        docs = await retriever_search(q, k=k)

    logger.info(f"Retrieved {len(docs)} documents")
    state["docs"] = docs
    return state
