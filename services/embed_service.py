"""임베딩 생성 및 벡터 저장 서비스"""
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from app.deps import get_openai_client, get_vectorstore, get_settings


BATCH_SIZE = 256  # 배치 크기 조절


async def _generate_embeddings(
    texts: List[str],
) -> List[List[float]]:
    """OpenAI API로 텍스트 임베딩 생성"""
    client = get_openai_client()
    model = get_settings().embed_model
    vectors: List[List[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        resp = await client.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in resp.data])

    return vectors


async def embed_and_store_documents(
    docs: List[Document],
    announcement_count: int,
) -> None:
    """
    문서들을 임베딩하여 벡터 DB에 저장

    Args:
        docs: 저장할 문서 리스트
        announcement_count: 공지사항 개수
    """
    texts = [doc.page_content for doc in docs]

    # 임베딩 생성
    vectors = await _generate_embeddings(texts=texts)

    # 벡터 스토어에 저장
    vector_store: PGVector = get_vectorstore()
    await vector_store.aadd_embeddings(
        texts=texts,
        metadatas=[doc.metadata for doc in docs],
        embeddings=vectors
    )
