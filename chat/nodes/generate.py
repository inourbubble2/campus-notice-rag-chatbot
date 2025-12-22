from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from app.deps import get_chat_llm
from chat.schema import RAGState

GEN_SYS = """당신은 서울시립대학교 공지사항 Q&A 도우미입니다.

지침:
1. **질문의 핵심 키워드를 포함하여** 자연스럽게 답변을 시작하세요. (예: "2025학년도 인향제 축제 기간은..." 와 같이 질문 내용을 반복)
2. **마크다운 문법(**, -, ~, # 등)은 절대 사용하지 마세요.** (텍스트로만 출력)
3. 제공된 공지사항 내용에 기반하여 정확한 사실만 전달하세요.
4. 오직 제공된 '이전 대화'와 '참고할 공지'에 있는 내용만 사용하여 사용자의 질문에 답변하세요. 외부 지식이나 추측을 사용하지 마세요.
5. 현재 년도는 2025년입니다.
6. 질문에 직접적인 답만 포함하세요. 인사말, 마무리 멘트, 잡담은 포함하지 마세요.
"""

GEN_USER_TMPL = """이전 대화:
{chat_history}

질문: {question}

재작성된 질의: {rewrite_result}

참고할 공지:
{context}

이 질문에 대해 정확한 답변을 작성하세요."""

gen_prompt = ChatPromptTemplate.from_messages([("system", GEN_SYS), ("user", GEN_USER_TMPL)])

def format_context(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        md = d.metadata or {}
        lines.append(
            f"  Doc Id: {md.get('announcement_id')}\n"
            f"  Doc: {d.page_content}"
        )
    return "\n".join(lines)

async def generate_node(state: RAGState, config: RunnableConfig) -> dict:
    context = format_context(state.docs)

    question = state.question

    messages = state.messages
    history_msgs = messages[:-1]
    history_str = "\n".join([f"- {m.type}: {m.content}" for m in history_msgs[-6:]])

    msgs = gen_prompt.format_messages(
        question=question,
        chat_history=history_str,
        rewrite_result=state.rewrite or {},
        context=context,
    )

    out = await get_chat_llm().ainvoke(msgs, config=config)

    return {"messages": [out], "answer": out.content}
