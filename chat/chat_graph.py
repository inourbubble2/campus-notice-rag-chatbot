# chat_graph.py
import json
import logging
from typing import TypedDict, List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.deps import get_small_llm, get_chat_llm
from services.retriever_service import retriever_search

logger = logging.getLogger(__name__)

# =========================
# State
# =========================
class RAGState(TypedDict):
    question: str
    chat_history: List[dict]
    docs: List[Document]
    answer: Optional[str]
    rewrite: Optional[Dict[str, Any]]  # {"query": str, "keywords": [...], "filters": {...}}
    attempt: int
    validate: Optional[Dict[str, Any]]  # {"decision": "PASS|RETRY", "reason": str, "critic_query": str|None}
    guardrail: Optional[Dict[str, Any]]  # {"policy": "PASS|BLOCK", "reason": str}

# =========================
# Guardrail Node
# =========================
GUARD_SYS = """당신은 대학 Q&A 서비스의 가드레일입니다.
아래 기준으로만 JSON을 반환하세요.
- 대학 공지/규칙/행사 등 대학과 관련 없는 내용, 개인식별정보 요구/제공, 법/의료/금융 고위험 조언, 혐오/폭력 조장, 성인물, 공격적/위법한 요청 → BLOCK
- 그 외 → PASS

반환 스키마:
{3
  "policy": "PASS" | "BLOCK",
  "reason": "간단한 한국어 설명"
}
설명 문장 없이 JSON만 출력하세요.
"""

GUARD_USER_TMPL = """사용자 질문:
{{ question }}"""

guard_prompt = ChatPromptTemplate.from_messages(
    [("system", GUARD_SYS), ("user", GUARD_USER_TMPL)],
    template_format="jinja2",
)

def guardrail_node(state: RAGState) -> RAGState:
    msgs = guard_prompt.format_messages(question=state["question"])
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

# =========================
# Rewrite Node
# =========================
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

REWRITE_USER_TMPL = """원 질문:
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
    msgs = rewrite_prompt.format_messages(question=state["question"])
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

# =========================
# Retrieve Node
# =========================
BASE_K = 6
K_STEP = 4
K_MAX = 20

async def retrieve_node(state: RAGState) -> RAGState:
    rw = state.get("rewrite") or {}
    q = rw.get("query") or state["question"]
    filters = rw.get("filters", {})

    # 시도 횟수에 따라 k 증가
    k = min(BASE_K + state.get("attempt", 0) * K_STEP, K_MAX)

    # 필터가 있는지 확인
    has_filters = any([
        filters.get("departments"),
        filters.get("grades"),
        filters.get("tags")
    ])

    if has_filters:
        logger.info(f"Retrieving documents with filters: k={k}, attempt={state.get('attempt', 0)}")
        docs = await retriever_search(q, k=k, filters=filters)
    else:
        logger.info(f"Retrieving documents: k={k}, attempt={state.get('attempt', 0)}")
        docs = await retriever_search(q, k=k)

    logger.info(f"Retrieved {len(docs)} documents")
    state["docs"] = docs
    return state

# =========================
# Generate Node
# =========================
GEN_SYS = """당신은 서울시립대학교 공지사항 Q&A 도우미입니다.
지침:
- 사용자의 질문에 ‘공지 원문 근거’를 바탕으로 한국어로 정확히 답변하세요.
- 날짜/마감일은 가능하면 'YYYY-MM-DD(요일)'로 표기하세요.
- 확실하지 않으면 모호함을 명시하고, 관련 공지를 2~3개 더 제안하세요.
- 마지막에 "근거" 섹션으로 참고한 공지의 ID/제목/작성일/URL을 bullet로 나열하세요.
- 공지의 문구를 그대로 복사하기보다 핵심만 요약하세요.
"""

GEN_USER_TMPL = """질문: {question}

검색 질의(재작성): {rewritten_query}
핵심 키워드: {keywords}
적용된 필터: {filters}

참고할 공지(최대 6개):
{context}

이 질문에 대해 정확한 답변을 작성하세요."""

gen_prompt = ChatPromptTemplate.from_messages([("system", GEN_SYS), ("user", GEN_USER_TMPL)])

def format_context(docs: List[Document]) -> str:
    lines = []
    for d in docs[:6]:
        md = d.metadata or {}

        # 기본 정보
        info_parts = [
            f"게시판:{md.get('board', '미상')}",
            f"학과:{md.get('major', '전체')}",
            f"작성일:{md.get('written_at', '미상')}",
            f"작성자:{md.get('author', '미상')}"
        ]

        # 구조화된 메타데이터 추가
        tags = md.get('tags', [])
        if tags:
            info_parts.append(f"태그:{','.join(tags)}")

        depts = md.get('target_departments', [])
        if depts:
            info_parts.append(f"대상학과:{','.join(depts)}")

        grades = md.get('target_grades', [])
        if grades:
            info_parts.append(f"대상학년:{','.join(map(str, grades))}학년")

        lines.append(
            f"- [{md.get('title','(제목없음)')}] ({', '.join(info_parts)})\n"
            f"  URL: {md.get('url', '없음')}\n"
            f"  본문요약용청크: {d.page_content[:350]}..."
        )
    return "\n".join(lines)

def generate_node(state: RAGState) -> RAGState:
    rw = state.get("rewrite") or {}
    ctx = format_context(state["docs"])

    # 필터 정보 포맷팅
    filters = rw.get("filters", {})
    filter_parts = []
    if filters.get("departments"):
        filter_parts.append(f"학과: {', '.join(filters['departments'])}")
    if filters.get("grades"):
        filter_parts.append(f"학년: {', '.join(map(str, filters['grades']))}학년")
    if filters.get("tags"):
        filter_parts.append(f"태그: {', '.join(filters['tags'])}")
    filter_str = ", ".join(filter_parts) if filter_parts else "없음"

    msgs = gen_prompt.format_messages(
        question=state["question"],
        rewritten_query=rw.get("query") or state["question"],
        keywords=", ".join(rw.get("keywords", [])),
        filters=filter_str,
        context=ctx,
    )
    out = get_chat_llm().invoke(msgs)

    # 출처 정리
    sources = []
    used = set()
    for d in state["docs"][:6]:
        m = d.metadata or {}
        key = (m.get("url"), m.get("title"))
        if key in used:
            continue
        used.add(key)
        sources.append(f"- {m.get('announcement_id')} | {m.get('title')} | {m.get('board')} | {m.get('major')} | {m.get('author')} | {m.get('written_at')} | {m.get('url')}")
    state["answer"] = out.content
    return state

# =========================
# Validate Node (answer quality gate)
# =========================
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
- 날짜/기한 등 핵심 정보가 근거 문서에 실제로 존재하는가? (추측/날조 금지)
- 근거 섹션에 최소 1개 이상의 URL/제목이 있는가?
- 표현이 모호하면 모호함을 명시했는가?
"""

VAL_USER_TMPL = """원 질문:
{{ question }}

생성된 답변:
{{ answer }}

문서 개수: {{ doc_count }}
문서 목록(제목들만):
{% for d in docs %}
- {{ d }}
{% endfor %}
"""

val_prompt = ChatPromptTemplate.from_messages(
    [("system", VAL_SYS), ("user", VAL_USER_TMPL)],
    template_format="jinja2",
)

def validate_node(state: RAGState) -> RAGState:
    titles = [ (d.metadata or {}).get("title","(제목없음)") for d in state.get("docs", []) ]
    msgs = val_prompt.format_messages(
        question=state["question"],
        answer=state.get("answer", ""),
        doc_count=len(state.get("docs", [])),
        docs=titles,
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

# =========================
# Graph Wiring (with loop)
# =========================
MAX_ATTEMPTS = 2  # 총 1차 시도 + 2번 재시도 = 최대 3회

graph = StateGraph(RAGState)

graph.add_node("guardrail", guardrail_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("validate", validate_node)

graph.add_edge(START, "guardrail")

# guardrail 결과에 따라 분기
def guardrail_router(state: RAGState):
    if (state.get("guardrail") or {}).get("policy") == "BLOCK":
        return END
    return "rewrite"

graph.add_conditional_edges("guardrail", guardrail_router, ["rewrite", END])
graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "validate")

# validate 결과에 따라 PASS면 종료, RETRY면 루프(재검색→재생성)
def validate_router(state: RAGState):
    if (state.get("validate") or {}).get("decision") == "RETRY" and state.get("attempt", 0) < MAX_ATTEMPTS:
        return "retrieve"
    return END

graph.add_conditional_edges("validate", validate_router, ["retrieve", END])

app = graph.compile(checkpointer=MemorySaver())
