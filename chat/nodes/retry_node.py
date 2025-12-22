from chat.schema import RAGState
from langchain_core.messages import HumanMessage

def retry_node(state: RAGState) -> dict:
    attempt = state.attempt + 1
    reason = "답변이 충분하지 않습니다."
    if state.validation:
        reason = state.validation.reason or reason
        critic = state.validation.critic_query
        if critic:
            reason += f" (제안 검색어: {critic})"
            
    # Add a message to prompt the agent to try again
    retry_msg = HumanMessage(content=f"검증 피드백: {reason}. 더 나은 정보를 검색하거나 답변을 다시 작성해주세요.")
    
    return {
        "attempt": attempt, 
        "messages": [retry_msg]
    }
