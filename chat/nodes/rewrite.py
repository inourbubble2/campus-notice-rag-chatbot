import json
import logging
from langchain.prompts import ChatPromptTemplate
from app.deps import get_small_llm
from chat.schema import RAGState

logger = logging.getLogger(__name__)

REWRITE_SYS = """당신은 대학 공지사항 RAG를 위한 질의 재작성 및 필터 추출기입니다.
JSON 한 객체만 출력하고 설명/문장/코드블록은 금지합니다.
스키마:
{% raw %}
{
  "query": "핵심 키워드 위주의 간결한 문장 (조사/어미 최소화)",
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
1. **query**:
   - **핵심 키워드와 명사 위주로 간결하게 작성하세요.** (문장 형태보다 키워드 나열이 검색에 유리할 수 있습니다.)
   - **고유명사(행사명, 프로그램명, 장소 등)는 절대 변경하거나 생략하지 말고 그대로 유지하세요.**
   - 불필요한 조사나 어미를 제거하고, 동의어를 괄호로 추가하여 검색 범위를 넓히세요. (예: 휴학(휴학신청, 휴학기간))
2. **keywords**: 핵심 키워드 추출 (너무 일반적인 단어 제외)

반드시 유효한 JSON 한 객체만 출력하세요.
"""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [("system", REWRITE_SYS), ("user", REWRITE_USER_TMPL)],
    template_format="jinja2",
)

def rewrite_node(state: RAGState) -> RAGState:
    msgs = rewrite_prompt.format_messages(
        question=state["question"],
        chat_history=state.get("chat_history", [])
    )
    out = get_small_llm().invoke(msgs)

    data = json.loads(out.content.strip())
    if not isinstance(data.get("query", ""), str) or not data["query"].strip():
        data["query"] = state["question"]
    if not isinstance(data.get("query", ""), str) or not data["query"].strip():
        data["query"] = state["question"]
    if not isinstance(data.get("keywords", []), list):
        data["keywords"] = []

    data = {
        "query": data["query"],
        "keywords": data["keywords"]
    }

    state["rewrite"] = data
    # 초기 시도 카운터
    state["attempt"] = state.get("attempt", 0) or 0

    # 로그 출력
    logger.info(f"Query rewritten: '{state['question']}' -> '{data.get('query')}'")
    logger.info(f"Keywords: {data.get('keywords')}")

    return state
