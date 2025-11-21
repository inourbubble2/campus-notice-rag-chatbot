from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    chat_history: List[dict]
    docs: List[Document]
    answer: Optional[str]
    rewrite: Optional[Dict[str, Any]]  # {"query": str, "keywords": [...], "filters": {...}}
    attempt: int
    validate: Optional[Dict[str, Any]]  # {"decision": "PASS|RETRY", "reason": str, "critic_query": str|None}
    guardrail: Optional[Dict[str, Any]]  # {"policy": "PASS|BLOCK", "reason": str}
