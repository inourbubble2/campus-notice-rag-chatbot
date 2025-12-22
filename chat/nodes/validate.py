import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from app.deps import get_small_llm
from chat.schema import RAGState, ValidateResult

logger = logging.getLogger(__name__)

VAL_SYS = """당신은 공지사항 RAG의 품질 검증기입니다.
현재 제공된 공지사항으로 답변할 수 없다고 말한다면 RETRY를 권장합니다.

판단 기준(하나라도 충족 못하면 RETRY 권장):
- 답변이 질문에 직접적으로 응답하는가?
- 표현이 모호하면 모호함을 명시했는가?
"""

VAL_USER_TMPL = """원 질문:
{{ question }}

생성된 답변:
{{ answer }}

문서 목록:
{{ docs }}
"""

val_prompt = ChatPromptTemplate.from_messages(
    [("system", VAL_SYS), ("user", VAL_USER_TMPL)],
    template_format="jinja2",
)

def validate_node(state: RAGState, config: RunnableConfig) -> dict:
    page_contents = [ d.page_content for d in state.docs ]
    docs_str = "\n".join([f"- {content}" for content in page_contents])
    
    question = state.question
    answer = state.answer

    msgs = val_prompt.format_messages(
        question=question,
        answer=answer,
        docs=docs_str,
    )

    structured_llm = get_small_llm().with_structured_output(ValidateResult)

    result: ValidateResult = structured_llm.invoke(msgs, config=config)

    return {"validation": result}
