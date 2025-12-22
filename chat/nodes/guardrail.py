
import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from app.deps import get_small_llm
from chat.schema import RAGState, GuardrailResult

logger = logging.getLogger(__name__)

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

def guardrail_node(state: RAGState, config: RunnableConfig) -> dict:
    # Messages
    messages = state.messages
    question = messages[-1].content if messages else ""
    
    # History (everything except the last message)
    history_msgs = messages[:-1]
    history_str = "\n".join([f"- {m.type}: {m.content}" for m in history_msgs[-6:]])

    msgs = guard_prompt.format_messages(
        question=question,
        chat_history=history_str
    )

    structured_llm = get_small_llm().with_structured_output(GuardrailResult)

    result: GuardrailResult = structured_llm.invoke(msgs, config=config)
    
    return {
        "guardrail": result,
        "question": question
    }
