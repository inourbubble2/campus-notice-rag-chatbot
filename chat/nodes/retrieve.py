import logging
from services.retriever_service import retriever_search
from chat.schema import RAGState

logger = logging.getLogger(__name__)

BASE_K = 6
K_STEP = 4
K_MAX = 20

async def retrieve_node(state: RAGState) -> dict:
    query = state.question
    if state.rewrite and state.rewrite.query:
        query = state.rewrite.query

    k = min(BASE_K + state.attempt * K_STEP, K_MAX)

    docs = await retriever_search(query, k)

    return {"docs": docs}
