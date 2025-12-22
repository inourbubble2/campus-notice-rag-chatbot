import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from app.deps import get_small_llm
from chat.schema import RAGState, RewriteResult

logger = logging.getLogger(__name__)

REWRITE_SYS = """당신은 '대학 공지사항 RAG 시스템'을 위한 질의 재작성기(쿼리 리라이터)입니다.

역할:
- 사용자의 자연어 질문과 이전 대화를 기반으로,
  벡터 검색 및 키워드 검색에 모두 잘 걸리는 형태로 질의를 재작성합니다.
- 답변을 생성하는 것이 아니라, 검색용 쿼리(query)를 만드는 것이 목적입니다.

도메인:
- 대학(특히 서울시립대학교)의 공지사항, 학사 안내, 장학/프로그램, 행사, 튜터링, 학점교류, 휴학/복학 등과 관련된 정보를 다룹니다.

원칙:
1. 의미는 보존하되, 공지 제목처럼 간결하고 정보 밀도가 높은 표현으로 바꿉니다.
2. 대학/학사 맥락(학교명, 학년도, 학기, 전공/학부, 프로그램명 등)이 드러나면 검색 품질이 높아지므로, 질문에 언급된 정보는 가능한 한 유지·명시합니다.
3. 모호한 대명사(이것, 저것, 거기, 그때 등)는 chat_history를 참고해 가능한 한 구체적인 명사(과목명, 프로그램명, 행사명 등)로 치환합니다.
4. 새로운 사실을 지어내거나, 질문에 없는 구체적인 날짜·조건을 임의로 추가하지 않습니다.
"""

REWRITE_USER_TMPL = """대화 기록:
{{ chat_history }}

원 질문:
{{ question }}

지침:
- 사용자의 질문과 이전 대화 맥락의 의미를 보존하면서, 공지 제목/검색어처럼 간결하게 재작성합니다.
- 불필요한 구어체(나, 좀, 언제야, 알려줘 등)와 의미 없는 불용어는 제거합니다.
- 대학/학사 도메인 맥락이 드러나도록, 학교명·학년도·학기·전공/학부·프로그램명·행사명을 가능한 한 명시합니다.
- 유사 표현·동의어·변형어는 괄호를 사용해 보강합니다.
 예: 휴학(휴학신청, 휴학기간), 복수전공(이중전공), 기숙사(국제학사, 하계 기숙사)
- 최종 형태는 공지 제목 같은 한 문장으로 작성합니다.
 예: "2025학년도 2학기 시대튜터링 학습도우미 지원 자격 및 평점 기준 안내"
"""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [("system", REWRITE_SYS), ("user", REWRITE_USER_TMPL)],
    template_format="jinja2",
)


def rewrite_node(state: RAGState, config: RunnableConfig) -> dict:
    question = state.question

    messages = state.messages
    history_msgs = messages[:-1]
    history_str = "\n".join([f"- {m.type}: {m.content}" for m in history_msgs[-6:]])

    msgs = rewrite_prompt.format_messages(
        question=question,
        chat_history=history_str
    )

    structured_llm = get_small_llm().with_structured_output(RewriteResult)

    result: RewriteResult = structured_llm.invoke(msgs, config=config)

    return {"rewrite": result, "attempt": state.attempt}
