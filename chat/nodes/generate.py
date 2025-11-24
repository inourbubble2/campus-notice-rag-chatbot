from typing import List
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from app.deps import get_chat_llm
from chat.schema import RAGState

GEN_SYS = """당신은 서울시립대학교 공지사항 Q&A 도우미입니다.
지침:
- 현재 날짜는 2025년입니다.
- 오직 제공된 '이전 대화'와 '참고할 공지'에 있는 내용만 사용하여 사용자의 질문에 답변하세요. 외부 지식이나 추측을 사용하지 마세요.
- 날짜/마감일은 'YYYY-MM-DD(요일)'로 명확히 표기하세요.
- 답변에 마크다운 문법 **, ~, 등을 사용하지 마세요.
- 자연스러운 톤으로 대답하세요.
"""

GEN_USER_TMPL = """이전 대화:
{chat_history}

질문: {question}

검색 질의(재작성): {rewritten_query}
핵심 키워드: {keywords}

참고할 공지:
{context}

이 질문에 대해 정확한 답변을 작성하세요."""

gen_prompt = ChatPromptTemplate.from_messages([("system", GEN_SYS), ("user", GEN_USER_TMPL)])

def format_context(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        md = d.metadata or {}
        lines.append(
            f"- Title: [{md.get('title','(제목없음)')}]\n"
            f"  Doc Id: {md.get('announcement_id')}\n"
            f"  본문: {d.page_content}"
        )
    return "\n".join(lines)

async def generate_node(state: RAGState) -> RAGState:
    rw = state.get("rewrite") or {}
    ctx = format_context(state["docs"])

    # Chat history formatting
    history = state.get("chat_history", [])
    history_str = "\n".join([f"- {m['role']}: {m['content']}" for m in history[-6:]])

    msgs = gen_prompt.format_messages(
        question=state["question"],
        chat_history=history_str,
        rewritten_query=rw.get("query") or state["question"],
        keywords=", ".join(rw.get("keywords", [])),
        context=ctx,
    )

    # Generate answer
    out = get_chat_llm().invoke(msgs)

    # 출처 정리
    sources = []
    used = set()
    for d in state["docs"]:
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
