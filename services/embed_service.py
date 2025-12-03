"""임베딩 생성 및 벡터 저장 서비스"""
from typing import List
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from app.deps import get_openai_client, get_vectorstore, get_settings
import logging

logger = logging.getLogger(__name__)


BATCH_SIZE = 256  # 배치 크기 조절


async def _generate_embeddings(
    texts: List[str],
) -> List[List[float]]:
    client = get_openai_client()
    model = get_settings().embed_model
    vectors: List[List[float]] = []

    total_tokens = 0
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        resp = await client.embeddings.create(model=model, input=batch)

        vectors.extend(d.embedding for d in resp.data)
        total_tokens += getattr(resp.usage, "total_tokens", 0)

    logger.info(f"Embedding generated for {len(texts)} texts. Total Token Usage: {total_tokens}")
    return vectors


async def embed_and_store_documents(
    docs: List[Document],
) -> None:
    texts = [doc.page_content for doc in docs]

    vectors = await _generate_embeddings(texts=texts)
    vector_store: PGVector = get_vectorstore()

    await vector_store.aadd_embeddings(
        texts=texts,
        metadatas=[doc.metadata for doc in docs],
        embeddings=vectors
    )
