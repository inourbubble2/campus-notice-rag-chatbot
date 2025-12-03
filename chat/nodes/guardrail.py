
import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from app.deps import get_small_llm
from chat.schema import RAGState

logger = logging.getLogger(__name__)

from typing import Literal
from pydantic import BaseModel, Field

class GuardrailResult(BaseModel):
    """가드레일 판정 결과"""
    policy: Literal["PASS", "BLOCK"] = Field(description="판정 결과 (PASS 또는 BLOCK)")
    reason: str = Field(description="판정 이유 (간단한 한국어 설명)")

GUARD_SYS = """당신은 대학 Q&A 서비스의 가드레일입니다.
아래 기준으로 판정하세요.
- 대학 공지/규칙/행사 등 대학과 관련 없는 내용, 혐오/폭력 조장, 성인물, 공격적/위법한 요청 → BLOCK
- 단, **이전 대화 맥락(Context)에서 이어지는 질문**이라면, 겉보기에 대학과 무관해 보여도 허용(PASS)하세요. (예: "그거 언제야?", "어디서 해?" 등)
- 그 외 → PASS
"""

GUARD_USER_TMPL = """대화 기록:
{{ chat_history }}

사용자 질문:
{{ question }}
"""

guard_prompt = ChatPromptTemplate.from_messages(
    [("system", GUARD_SYS), ("user", GUARD_USER_TMPL)],
    template_format="jinja2",
)

def guardrail_node(state: RAGState, config: RunnableConfig) -> RAGState:
    # Chat history formatting
    history = state.get("chat_history", [])
    history_str = "\n".join([f"- {m['role']}: {m['content']}" for m in history[-6:]])

    msgs = guard_prompt.format_messages(
        question=state["question"],
        chat_history=history_str
    )

    structured_llm = get_small_llm().with_structured_output(GuardrailResult)

    try:
        result: GuardrailResult = structured_llm.invoke(msgs, config=config)
        state["guardrail"] = {"policy": result.policy, "reason": result.reason}
    except Exception as e:
        logger.warning(f"Guardrail failed: {e}")
        state["guardrail"] = {"policy": "PASS", "reason": ""}

    return state
