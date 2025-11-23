import asyncio
import logging
from langchain_core.documents import Document
from services.retriever_service import rerank_documents

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_reranker():
    print("Testing Reranker (Dongjin-kr/ko-reranker)...")
    
    # 더미 문서 생성
    docs = [
        Document(page_content="이것은 테스트 문서입니다.", metadata={"id": 1}),
        Document(page_content="서울시립대학교 공지사항입니다.", metadata={"id": 2}),
        Document(page_content="파이썬 프로그래밍 강의", metadata={"id": 3}),
        Document(page_content="2024학년도 2학기 장학금 신청 안내", metadata={"id": 4}),
        Document(page_content="도서관 이용 시간 변경", metadata={"id": 5}),
    ]
    
    query = "장학금 신청 기간"
    print(f"Query: {query}")
    
    # 리랭킹 수행
    reranked_docs = rerank_documents(query, docs, top_n=3)
    
    print("\n[Reranked Results]")
    for i, doc in enumerate(reranked_docs):
        print(f"{i+1}. Score: {doc.metadata.get('score', 'N/A'):.4f} - {doc.page_content}")

    # 검증: '장학금' 관련 문서가 1위여야 함
    if "장학금" in reranked_docs[0].page_content:
        print("\nSUCCESS: Reranker prioritized relevant document.")
    else:
        print("\nFAILURE: Reranker did not prioritize relevant document.")

if __name__ == "__main__":
    asyncio.run(test_reranker())
