
import logging
from chat.schema import RAGState, RewriteResult

logger = logging.getLogger(__name__)

def refine_query_node(state: RAGState) -> dict:
    query = f"{state.question} | {state.rewrite.query} | {state.validation.critic_query}"

    return {
        "rewrite": RewriteResult(query=query),
        "attempt": state.attempt + 1
    }
