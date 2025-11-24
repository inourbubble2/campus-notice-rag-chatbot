# services/retriever_service.py
"""
벡터 스토어 검색 서비스.
"""
import logging
from typing import List
from langchain_core.documents import Document

from app.deps import get_vectorstore

logger = logging.getLogger(__name__)


async def retriever_search(
    query: str,
    k: int,
) -> List[Document]:
    vectorstore = get_vectorstore()

    docs_with_score = await vectorstore.asimilarity_search_with_score(query, k)
    docs = []
    for doc, score in docs_with_score:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["score"] = score
        docs.append(doc)

    return docs


if __name__ == "__main__":
    import asyncio

    async def main():
        # test
        test_query = "미래디자인"
        print(f"Query: {test_query}\n")

        results = await retriever_search(query=test_query, k=5)

        print(f"Found {len(results)} documents:\n")
        for i, doc in enumerate(results, 1):
            print(f"=== Document {i} ===")
            print(f"Content: {doc.page_content}...")
            print(f"Metadata: {doc.metadata}")
            print()

    asyncio.run(main())
