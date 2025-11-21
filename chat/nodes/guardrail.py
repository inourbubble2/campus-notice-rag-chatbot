import json
import logging
from langchain.prompts import ChatPromptTemplate
from app.deps import get_small_llm
from chat.schema import RAGState

logger = logging.getLogger(__name__)

GUARD_SYS = """당신은 대학 Q&A 서비스의 가드레일입니다.
아래 기준으로만 JSON을 반환하세요.
- 대학 공지/규칙/행사 등 대학과 관련 없는 내용, 혐오/폭력 조장, 성인물, 공격적/위법한 요청 → BLOCK
- 단, **이전 대화 맥락(Context)에서 이어지는 질문**이라면, 겉보기에 대학과 무관해 보여도 허용(PASS)하세요. (예: "그거 언제야?", "어디서 해?" 등)
- 그 외 → PASS

반환 스키마:
{
  "policy": "PASS" | "BLOCK",
  "reason": "간단한 한국어 설명"
}
설명 문장 없이 JSON만 출력하세요.
"""

GUARD_USER_TMPL = """대화 기록:
{% for msg in chat_history[-6:] %}
- {{ msg.role }}: {{ msg.content }}
{% endfor %}

사용자 질문:
{{ question }}"""

guard_prompt = ChatPromptTemplate.from_messages(
    [("system", GUARD_SYS), ("user", GUARD_USER_TMPL)],
    template_format="jinja2",
)

def guardrail_node(state: RAGState) -> RAGState:
    msgs = guard_prompt.format_messages(
        question=state["question"],
        chat_history=state.get("chat_history", [])
    )
    out = get_small_llm().invoke(msgs)
    try:
        data = json.loads(out.content)
        pol = data.get("policy", "PASS")
        if pol not in ("PASS", "BLOCK"):
            pol = "PASS"
        state["guardrail"] = {"policy": pol, "reason": data.get("reason", "")}
    except Exception as e:
        logger.warning(f"Guardrail JSON parse failed: {e}. LLM output: {out.content[:200]}")
        state["guardrail"] = {"policy": "PASS", "reason": ""}
    return state
