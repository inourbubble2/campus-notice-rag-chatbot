# app/deps.py
"""
FastAPI에서 재사용할 공용 의존성 모듈.
- Settings: 환경 변수 관리
- SQLAlchemy Engine
- OpenAI Embeddings / Chat LLM
- PGVector VectorStore / Retriever
모두 lazy singleton으로 초기화됩니다.
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

load_dotenv()


# ---------- Settings ----------
class Settings(BaseSettings):
  # API Keys
  openai_api_key: str

  # DB / Vector
  pg_conn: str
  collection_name: str = "uos_announcement"
  embed_model: str = "text-embedding-3-small"
  embed_dim: int = 1536
  use_jsonb: bool = True

  # LLM
  chat_model: str = "gpt-5-mini"        # 최종 응답 생성용
  small_model: str = "gpt-5-mini"       # 가벼운 재작성/가드/검증용
  temperature: float = 0.0
  request_timeout: int = 60             # seconds

  # Retriever 기본값
  retriever_k: int = 6
  retriever_fetch_k: int = 40
  retriever_mmr: bool = True

  class Config:
    env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
  """애플리케이션 전역 설정 (싱글톤 캐시)."""
  return Settings()


# ---------- Database ----------
_engine: Optional[Engine] = None

def get_engine() -> Engine:
  """SQLAlchemy Engine (lazy singleton)."""
  global _engine
  if _engine is None:
    cfg = get_settings()
    _engine = create_engine(cfg.pg_conn, pool_pre_ping=True)
  return _engine


# ---------- Embeddings ----------
_embeddings: Optional[OpenAIEmbeddings] = None

def get_embeddings() -> OpenAIEmbeddings:
  """OpenAI 임베딩 인스턴스."""
  global _embeddings
  if _embeddings is None:
    cfg = get_settings()
    _embeddings = OpenAIEmbeddings(
        model=cfg.embed_model,
        api_key=cfg.openai_api_key
    )
  return _embeddings


# ---------- VectorStore (PGVector) ----------
_vectorstore: Optional[VectorStore] = None

def get_vectorstore() -> VectorStore:
  """PGVector VectorStore. 필요 시 pgvector 확장을 생성(create_extension=True)."""
  global _vectorstore
  if _vectorstore is None:
    cfg = get_settings()
    _vectorstore = PGVector(
        embeddings=get_embeddings(),
        connection=cfg.pg_conn,
        collection_name=cfg.collection_name,
        embedding_length=cfg.embed_dim,  # 인덱스 차원 명시
        use_jsonb=cfg.use_jsonb,
        create_extension=True,           # pgvector 확장 자동 생성(권장)
    )
  return _vectorstore


# ---------- Retriever ----------
def get_retriever(k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    mmr: Optional[bool] = None) -> BaseRetriever:
  """
  PGVector에서 Retriever 생성.
  - k: 최종 반환 개수
  - fetch_k: 후보 풀 크기(MMR/리랭크 전 풀)
  - mmr: 다양성(중복 억제)
  """
  cfg = get_settings()
  k = k if k is not None else cfg.retriever_k
  fetch_k = fetch_k if fetch_k is not None else cfg.retriever_fetch_k
  mmr = mmr if mmr is not None else cfg.retriever_mmr

  # langchain_postgres의 PGVector는 as_retriever를 그대로 지원
  return get_vectorstore().as_retriever(
      search_kwargs={
        "k": k,
        "fetch_k": fetch_k,
        "mmr": mmr,
      }
  )


# ---------- LLMs ----------
_chat_llm: Optional[ChatOpenAI] = None
_small_llm: Optional[ChatOpenAI] = None

def get_chat_llm() -> ChatOpenAI:
  """최종 답변 생성용 LLM."""
  global _chat_llm
  if _chat_llm is None:
    cfg = get_settings()
    _chat_llm = ChatOpenAI(
        model=cfg.chat_model,
        temperature=cfg.temperature,
        timeout=cfg.request_timeout,
        api_key=cfg.openai_api_key,
    )
  return _chat_llm

def get_small_llm() -> ChatOpenAI:
  """가벼운 작업(가드레일/재작성/검증)용 LLM."""
  global _small_llm
  if _small_llm is None:
    cfg = get_settings()
    _small_llm = ChatOpenAI(
        model=cfg.small_model,
        temperature=0,
        timeout=cfg.request_timeout,
        api_key=cfg.openai_api_key,
    )
  return _small_llm
