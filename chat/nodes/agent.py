import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
from app.deps import get_small_llm
from chat.schema import RAGState
from chat.tools import retreat_announcements

logger = logging.getLogger(__name__)

AGENT_SYS = """당신은 '대학 공지사항 검색 에이전트'입니다.
질문에 답하기 위해 검색도구(retreat_announcements)를 적극적으로 사용하세요.

[검색 쿼리 작성 원칙]
도구를 호출할 때 `query` 인자는 단순한 키워드가 아니라, 검색이 잘 되는 "풀 센텐스" 혹은 "구체적인 명사형 구문"이어야 합니다.
1. **맥락 반영**: 대화 기록(Context)을 참고하여 대명사(그거, 저거)를 구체적인 명사로 치환하세요.
2. **동의어 확장**: 질문의 핵심 단어와 관련된 행정 용어/유의어를 포함하세요. (예: 기숙사 -> 생활관, 국제학사)
3. **상세화**: 학년도, 학기, 모집 기간 등 구체적인 조건을 명시하세요.

예시:
- "장학금 언제 줘?" -> "2025학년도 1학기 장학금 지급 일정 및 시기"
- "전과 하려면?" -> "전과 신청 자격 요건 및 신청 기간"
"""

# Bind tools to the model
tools = [retreat_announcements]
llm = get_small_llm()
model_with_tools = llm.bind_tools(tools)

agent = create_agent(llm, tools=tools)

async def agent_node(state: RAGState, config: RunnableConfig) -> dict:
    messages = state.messages

    all_messages = [SystemMessage(content=AGENT_SYS)] + messages

    response = await model_with_tools.ainvoke(all_messages, config=config)

    return {"messages": [response], "answer": response.content if not response.tool_calls else None}
