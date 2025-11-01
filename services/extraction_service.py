# services/extraction_service.py
"""LLM 기반 구조화된 정보 추출 서비스"""
from typing import Optional
import logging

from app.deps import get_chat_llm
from models.announcement_parsed import AnnouncementParsedInfo

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM_PROMPT = """
당신은 대학교 공지사항에서 중요한 정보를 추출하는 전문가입니다.
주어진 공지사항 텍스트를 분석하여 다음 정보를 추출해주세요:

**추출 항목:**
1. application_period_start: 신청/접수 시작 날짜 (YYYY-MM-DD 또는 YYYY-MM-DD HH:MM 형식)
2. application_period_end: 신청/접수 마감 날짜 (YYYY-MM-DD 또는 YYYY-MM-DD HH:MM 형식)
3. target_departments: 대상 학과 목록 (배열)
4. target_grades: 대상 학년 (1,2,3,4 중 해당하는 것들, 배열)
5. tags: 공지사항 카테고리 태그 (예: "장학금", "교환학생", "취업", "학사", "수업", "행사" 등)
6. additional_info: 추가적으로 중요한 정보 (자유 형식 딕셔너리)

**중요 규칙:**
- 명시적으로 언급되지 않은 정보는 null로 반환
- 날짜 형식을 정확히 맞출 것 (YYYY-MM-DD 또는 YYYY-MM-DD HH:MM)
- target_grades는 숫자 배열로 반환 (예: [1, 2, 3])
- "전체" 또는 "전 학년"이면 [1,2,3,4]로 반환
- "전 학과" 또는 "전체 학과"이면 target_departments를 null로 반환
- tags는 최대 5개까지만 선택
"""


def extract_structured_info(cleaned_text: str, title: str = "") -> Optional[AnnouncementParsedInfo]:
    """
    LLM을 사용해서 공지사항에서 구조화된 정보 추출.
    LangChain의 with_structured_output()을 사용하여 JSON 출력 강제화.

    Args:
        cleaned_text: HTML이 제거된 정제 텍스트 & OCR 결과
        title: 공지사항 제목

    Returns:
        StructuredAnnouncementInfo 객체 또는 None (실패 시)
    """
    llm = get_chat_llm()
    structured_llm = llm.with_structured_output(AnnouncementParsedInfo)

    user_prompt = f"""
    공지사항 제목: {title}
    
    공지사항 내용:
    {cleaned_text}
    
    위 공지사항에서 구조화된 정보를 추출해주세요.
    """

    result = structured_llm.invoke([
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ])

    return result
