from typing import List, Optional, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class RewriteResult(BaseModel):
    """질의 재작성 결과"""
    query: str = Field(description="벡터 검색에 적합한 1~2문장 한국어 질의(핵심 키워드/동의어 포함)")

class ValidateResult(BaseModel):
    """답변 품질 검증 결과"""
    decision: Literal["PASS", "RETRY"] = Field(description="검증 결과 (PASS 또는 RETRY)")
    reason: str = Field(description="판정 이유 (간단한 한국어 설명)")
    critic_query: Optional[str] = Field(None, description="재시도 시 개선된 검색 질의 제안 (없으면 None)")

class GuardrailResult(BaseModel):
    """가드레일 판정 결과"""
    policy: Literal["PASS", "BLOCK"] = Field(description="판정 결과 (PASS 또는 BLOCK)")
    reason: str = Field(description="판정 이유 (간단한 한국어 설명)")

class RAGState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    question: str = Field(default="", description="The current user question being processed")
    
    docs: List[Document] = Field(default_factory=list)
    answer: Optional[str] = None
    
    # Nested Pydantic Models
    rewrite: Optional[RewriteResult] = None
    validation: Optional[ValidateResult] = None
    guardrail: Optional[GuardrailResult] = None
    
    attempt: int = 0
