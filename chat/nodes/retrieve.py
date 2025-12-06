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

    # 시도 횟수에 따라 k 증가
    k = min(BASE_K + state.get("attempt", 0) * K_STEP, K_MAX)

    docs = await retriever_search(q, k)

    for doc in docs:
        print(f"id: {doc.metadata.get('announcement_id')}, score:{doc.metadata.get('score')} - {doc.page_content[:150].replace(chr(10), ' ')}...")

    logger.info(f"Retrieved {len(docs)} documents")
    state["docs"] = docs
    return state
