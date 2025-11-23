from typing import List
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from app.deps import get_chat_llm
from chat.schema import RAGState

GEN_SYS = """당신은 서울시립대학교 공지사항 Q&A 도우미입니다.
지침:
- **오직 제공된 '참고할 공지'에 있는 내용만 사용하여 답변하세요.** 외부 지식이나 추측을 사용하지 마세요.
- **"제공된 공지에 따르면", "문서에 의하면" 같은 사족(Meta-talk)은 절대 사용하지 마세요.**
- 문맥에 답이 없으면 "제공된 공지사항 내용으로는 알 수 없습니다."라고 솔직하게 말하세요.
- 날짜/마감일은 'YYYY-MM-DD(요일)'로 명확히 표기하세요.
- 답변에 마크다운 문법 **, ~, 등을 사용하지 마세요.
- 사실에 기반하지 않은 내용은 절대 지어내지 마세요.
- 자연스러운 톤으로 대화하세요.
- 마지막에 추가 정보를 제시하세요.
"""

GEN_USER_TMPL = """이전 대화:
{chat_history}

질문: {question}

검색 질의(재작성): {rewritten_query}
핵심 키워드: {keywords}
적용된 필터: {filters}

참고할 공지:
{context}

이 질문에 대해 정확한 답변을 작성하세요."""

gen_prompt = ChatPromptTemplate.from_messages([("system", GEN_SYS), ("user", GEN_USER_TMPL)])

def format_context(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        md = d.metadata or {}

        # 기본 정보
        info_parts = [
            f"Doc ID:{md.get('announcement_id')}"
            f"학과:{md.get('major', '전체')}",
            f"작성일:{md.get('written_at', '미상')}",
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
            f"Title:[{md.get('title','(제목없음)')}]\n"
            f"URL: {md.get('url', '없음')}\n"
            f"Content: {d.page_content}\n"
            f"Info: {' | '.join(info_parts)}\n"
        )
    return "\n".join(lines)

async def generate_node(state: RAGState) -> RAGState:
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

    # Chat history formatting
    history = state.get("chat_history", [])
    history_str = "\n".join([f"- {m['role']}: {m['content']}" for m in history[-10:]])

    msgs = gen_prompt.format_messages(
        question=state["question"],
        chat_history=history_str,
        rewritten_query=rw.get("query") or state["question"],
        keywords=", ".join(rw.get("keywords", [])),
        filters=filter_str,
        context=ctx,
    )

    # Generate answer
    out = get_chat_llm().invoke(msgs)

    # 출처 정리
    sources = []
    used = set()
    for d in state["docs"][:15]:
        m = d.metadata or {}
        key = (m.get("url"), m.get("title"))
        if key in used:
            continue
        used.add(key)
        sources.append(f"- {m.get('announcement_id')} | {m.get('title')} | {m.get('board')} | {m.get('major')} | {m.get('author')} | {m.get('written_at')} | {m.get('url')}")

    state["answer"] = out.content

    # Update chat history
    new_history = state.get("chat_history", []) + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": out.content}
    ]
    state["chat_history"] = new_history

    return state
