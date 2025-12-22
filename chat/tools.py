from langchain.tools import tool, ToolRuntime
from services.retriever_service import retriever_search

BASE_K = 6
K_STEP = 2
K_MAX = 12

@tool
async def retreat_announcements(
    query: str,
    runtime: ToolRuntime
) -> str:
    """
    Search university announcements. Use this tool when you need information to answer the user's question.

    Args:
        query (str): Keywords or sentences to search in Korean
    """
    current_attempt = runtime.state.attempt
    k = min(BASE_K + current_attempt * K_STEP, K_MAX)

    docs = await retriever_search(query, k=k)

    formatted = "\n\n".join(
        [f"Document {i+1}:\n{doc.page_content}\nSource: {doc.metadata.get('url', 'N/A')}"
         for i, doc in enumerate(docs)]
    )

    return formatted
