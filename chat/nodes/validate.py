import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from app.deps import get_small_llm
from chat.schema import RAGState

logger = logging.getLogger(__name__)

from typing import Literal, Optional
from pydantic import BaseModel, Field

class ValidateResult(BaseModel):
    """답변 품질 검증 결과"""
    decision: Literal["PASS", "RETRY"] = Field(description="검증 결과 (PASS 또는 RETRY)")
    reason: str = Field(description="판정 이유 (간단한 한국어 설명)")
    critic_query: Optional[str] = Field(None, description="재시도 시 개선된 검색 질의 제안 (없으면 None)")

VAL_SYS = """당신은 공지사항 RAG의 품질 검증기입니다.

판단 기준(하나라도 충족 못하면 RETRY 권장):
- 답변이 질문에 직접적으로 응답하는가?
- 표현이 모호하면 모호함을 명시했는가?
"""

VAL_USER_TMPL = """원 질문:
{{ question }}

생성된 답변:
{{ answer }}

문서 목록:
{{ docs }}
"""

val_prompt = ChatPromptTemplate.from_messages(
    [("system", VAL_SYS), ("user", VAL_USER_TMPL)],
    template_format="jinja2",
)

def validate_node(state: RAGState, config: RunnableConfig) -> RAGState:
    page_contents = [ d.page_content for d in state.get("docs", []) ]
    docs_str = "\n".join([f"- {content}" for content in page_contents])

    msgs = val_prompt.format_messages(
        question=state["question"],
        answer=state.get("answer", ""),
        docs=docs_str,
    )

    structured_llm = get_small_llm().with_structured_output(ValidateResult)

    try:
        result: ValidateResult = structured_llm.invoke(msgs, config=config)
        state["validate"] = {
            "decision": result.decision,
            "reason": result.reason,
            "critic_query": result.critic_query or ""
        }
    except Exception as e:
        logger.warning(f"Validate failed: {e}")
        state["validate"] = {"decision": "PASS", "reason": "", "critic_query": ""}

    # 만약 RETRY라면, critic_query로 rewrite를 살짝 보강
    if state["validate"]["decision"] == "RETRY":
        logger.info(f"Validation failed, retrying: {state['validate'].get('reason')}")
        critic_q = state["validate"].get("critic_query") or ""
        if critic_q:
            # 기존 재작성 질의에 critic 힌트를 덧붙임
            rw = state.get("rewrite") or {"query": state["question"]}
            rw["query"] = f"{rw.get('query','')} | {critic_q}"
            state["rewrite"] = rw
            logger.info(f"Critic query added: {critic_q}")
        state["attempt"] = state.get("attempt", 0) + 1
    else:
        logger.info("Validation passed")

    return state
