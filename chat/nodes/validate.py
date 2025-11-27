import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from app.deps import get_small_llm
from chat.schema import RAGState

logger = logging.getLogger(__name__)

VAL_SYS = """당신은 공지사항 RAG의 품질 검증기입니다.
반드시 아래 JSON 하나만 반환하세요.
스키마:
{% raw %}
{
  "decision": "PASS" | "RETRY",
  "reason": "간단한 한국어 설명",
  "critic_query": "재시도 시 개선된 검색 질의 제안(없으면 빈 문자열)"
}
{% endraw %}

판단 기준(하나라도 충족 못하면 RETRY 권장):
- 답변이 질문에 직접적으로 응답하는가?
- 표현이 모호하면 모호함을 명시했는가?
"""

VAL_USER_TMPL = """원 질문:
{{ question }}

생성된 답변:
{{ answer }}

문서 목록:
{% for d in docs %}
- {{ d }}
{% endfor %}
"""

val_prompt = ChatPromptTemplate.from_messages(
    [("system", VAL_SYS), ("user", VAL_USER_TMPL)],
    template_format="jinja2",
)

def validate_node(state: RAGState) -> RAGState:
    page_contents = [ d.page_content for d in state.get("docs", []) ]
    msgs = val_prompt.format_messages(
        question=state["question"],
        answer=state.get("answer", ""),
        docs=page_contents,
    )
    out = get_small_llm().invoke(msgs)
    try:
        data = json.loads(out.content)
        decision = data.get("decision","PASS")
        if decision not in ("PASS","RETRY"):
            decision = "PASS"
        critic = data.get("critic_query") or ""
        state["validate"] = {"decision": decision, "reason": data.get("reason",""), "critic_query": critic}
    except Exception as e:
        logger.warning(f"Validate JSON parse failed: {e}. LLM output: {out.content[:200]}")
        state["validate"] = {"decision": "PASS", "reason": "", "critic_query": ""}

    # 만약 RETRY라면, critic_query로 rewrite를 살짝 보강
    if state["validate"]["decision"] == "RETRY":
        logger.info(f"Validation failed, retrying: {state['validate'].get('reason')}")
        critic_q = state["validate"].get("critic_query") or ""
        if critic_q:
            # 기존 재작성 질의에 critic 힌트를 덧붙임
            rw = state.get("rewrite") or {"query": state["question"], "keywords": []}
            rw["query"] = f"{rw.get('query','')} | {critic_q}"
            state["rewrite"] = rw
            logger.info(f"Critic query added: {critic_q}")
        state["attempt"] = state.get("attempt", 0) + 1
    else:
        logger.info("Validation passed")

    return state
