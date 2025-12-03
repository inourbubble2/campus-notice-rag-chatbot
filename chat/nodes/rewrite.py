import json
import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from app.deps import get_small_llm
from chat.schema import RAGState

logger = logging.getLogger(__name__)

REWRITE_SYS = """당신은 대학 공지사항 RAG를 위한 질의 재작성기입니다.
JSON 한 객체만 출력하고 설명/문장/코드블록은 금지합니다.
이전 대화와 원 질문을 참고하여 적절한 질의를 재작성하세요.
스키마:
{% raw %}
{
  "query": "벡터 검색에 적합한 1~2문장 한국어 질의(핵심 키워드/동의어 포함)",
  "keywords": ["핵심 키워드", "..."]
}
{% endraw %}
"""

REWRITE_USER_TMPL = """대화 기록:
{% for msg in chat_history[-6:] %}
- {{ msg.role }}: {{ msg.content }}
{% endfor %}

원 질문:
{{ question }}

지침:
1. **query**: 의미 보존하면서 불용어 제거, 동의어/변형어를 괄호로 보강 (예: 휴학(휴학신청,휴학기간))
2. **keywords**: 핵심 키워드 추출 (너무 일반적인 단어 제외, 고유명사가 있다면 추출)

반드시 유효한 JSON 한 객체만 출력하세요.
"""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [("system", REWRITE_SYS), ("user", REWRITE_USER_TMPL)],
    template_format="jinja2",
)


def rewrite_node(state: RAGState, config: RunnableConfig) -> RAGState:
    msgs = rewrite_prompt.format_messages(
        question=state["question"],
        chat_history=state.get("chat_history", [])
    )
    out = get_small_llm().invoke(msgs, config=config)
    try:
        data = json.loads(out.content.strip())
        if not isinstance(data.get("query", ""), str) or not data["query"].strip():
            data["query"] = state["question"]
        if not isinstance(data.get("keywords", []), list):
            data["keywords"] = []
    except Exception as e:
        logger.warning(f"Rewrite JSON parse failed: {e}. LLM output: {out.content[:200]}")
        data = {
            "query": state["question"],
            "keywords": [],
        }

    state["rewrite"] = data
    state["attempt"] = state.get("attempt", 0) or 0

    logger.info(f"Query rewritten: '{state['question']}' -> '{data.get('query')}'")
    logger.info(f"Keywords: {data.get('keywords')}")

    return state
