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
  "query": "벡터 검색에 적합한 1~2문장 한국어 질의(핵심 키워드/동의어 포함)",
  "keywords": ["핵심 키워드", "..."],
  "filters": {
    "departments": ["학과명1", "학과명2"],  // 질문에서 명시된 학과. 없으면 빈 배열
    "grades": [1, 2, 3, 4],  // 질문에서 명시된 학년 (숫자). 없으면 빈 배열
    "tags": ["장학금", "취업", "교환학생"]  // 질문의 주제/카테고리. 없으면 빈 배열
  }
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
2. **keywords**: 핵심 키워드 추출 (너무 일반적인 단어 제외)
3. **filters** 추출:
   - departments: 질문에 명시된 학과명 (예: "조경학과", "컴퓨터과학부", "전자전기공학부")
   - grades: 질문에 명시된 학년 (1, 2, 3, 4 중)
   - tags: 질문의 주제 (예: "장학금", "취업", "교환학생", "프로그램", "행사", "공모전", "대회" 등)

**학과명 예시**: 조경학과, 컴퓨터과학부, 전자전기공학부, 경영학부, 경제학부, 국어국문학과, 영어영문학과 등
**태그 예시**: 장학금, 취업, 인턴십, 교환학생, 봉사, 공모전, 대회, 특강, 세미나, 프로그램, 행사, 수업, 학사

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
    try:
        data = json.loads(out.content.strip())
        if not isinstance(data.get("query", ""), str) or not data["query"].strip():
            data["query"] = state["question"]
        if not isinstance(data.get("keywords", []), list):
            data["keywords"] = []

        # filters 검증 및 기본값 설정
        filters = data.get("filters", {})
        if not isinstance(filters, dict):
            filters = {}

        # 각 필터 필드 검증
        if not isinstance(filters.get("departments", []), list):
            filters["departments"] = []
        if not isinstance(filters.get("grades", []), list):
            filters["grades"] = []
        else:
            # grades는 숫자여야 함
            filters["grades"] = [g for g in filters["grades"] if isinstance(g, int) and 1 <= g <= 4]
        if not isinstance(filters.get("tags", []), list):
            filters["tags"] = []

        data["filters"] = filters

    except Exception as e:
        logger.warning(f"Rewrite JSON parse failed: {e}. LLM output: {out.content[:200]}")
        data = {
            "query": state["question"],
            "keywords": [],
            "filters": {"departments": [], "grades": [], "tags": []}
        }

    state["rewrite"] = data
    # 초기 시도 카운터
    state["attempt"] = state.get("attempt", 0) or 0

    # 로그 출력
    filters_info = data.get("filters", {})
    filter_summary = []
    if filters_info.get("departments"):
        filter_summary.append(f"학과:{filters_info['departments']}")
    if filters_info.get("grades"):
        filter_summary.append(f"학년:{filters_info['grades']}")
    if filters_info.get("tags"):
        filter_summary.append(f"태그:{filters_info['tags']}")

    logger.info(f"Query rewritten: '{state['question']}' -> '{data.get('query')}'")
    logger.info(f"Keywords: {data.get('keywords')}")
    if filter_summary:
        logger.info(f"Filters: {', '.join(filter_summary)}")

    return state
