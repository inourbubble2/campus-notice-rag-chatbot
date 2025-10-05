"""임베딩 생성 및 벡터 저장 서비스"""
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from app.deps import get_openai_client, get_vectorstore, get_settings
from models.metrics import EmbeddingMetrics

BATCH_SIZE = 256  # 배치 크기 조절


async def _embed_texts_with_usage(
    texts: List[str],
) -> Tuple[List[List[float]], int]:
    """OpenAI API로 텍스트 임베딩 생성 및 토큰 사용량 추적"""
    client = get_openai_client()
    model = get_settings().embed_model
    vectors: List[List[float]] = []
    total_tokens = 0

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        resp = await client.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in resp.data])
        if resp.usage:
            total_tokens += resp.usage.total_tokens or 0

    return vectors, total_tokens


async def embed_and_store_documents(
    docs: List[Document],
    announcement_count: int,
) -> EmbeddingMetrics:
    """
    문서들을 임베딩하여 벡터 DB에 저장하고 메트릭 반환

    Args:
        docs: 저장할 문서 리스트
        vector_store: PGVector 벡터 스토어
        model: 임베딩 모델 이름
        client: AsyncOpenAI 클라이언트
        announcement_count: 공지사항 개수

    Returns:
        EmbeddingMetrics: 임베딩 메트릭 (토큰 사용량, 비용 등)
    """
    texts = [doc.page_content for doc in docs]

    # 임베딩 생성 및 사용량 추적
    vectors, total_tokens = await _embed_texts_with_usage(texts=texts)

    # 벡터 스토어에 저장
    vector_store: PGVector = get_vectorstore()
    await vector_store.aadd_embeddings(
        texts=texts,
        metadatas=[doc.metadata for doc in docs],
        embeddings=vectors
    )

    return EmbeddingMetrics(
        announcement_count=announcement_count,
        document_chunks_count=len(docs),
        total_tokens=total_tokens,
    )
