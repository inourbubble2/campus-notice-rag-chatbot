from __future__ import annotations
from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
  # API Keys
  openai_api_key: str
  gemini_api_key: str
  upstage_api_key: str

  # DB / Vector
  pg_conn: str
  collection_name: str = "uos_announcement"
  embed_model: str = "text-embedding-3-small"
  embed_dim: int = 1536
  chunk_size: int = 1024
  chunk_overlap: int = 128
  use_jsonb: bool = True

  # LLM
  # LLM
  chat_model: str = "gpt-4o-mini"
  chat_model_provider: str = "openai"
  small_model: str = "gpt-4o-mini"
  small_model_provider: str = "openai"
  vision_model: str = "gemini-2.5-flash"
  vision_model_provider: str = "google_genai"
  temperature: float = 0.0
  llm_timeout: int = 60              # seconds
  small_llm_timeout: int = 5

  # OCR
  ocr_provider: str = "gemini"
  ocr_timeout: float = 120.0

  # Retriever 기본값
  retriever_k: int = 6
  retriever_fetch_k: int = 40
  retriever_mmr: bool = False  # MMR 활성화 (중복 제거)
  retriever_lambda_mult: float = 0.5  # MMR lambda: 0=다양성 우선, 1=유사도 우선

  class Config:
    env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
  """애플리케이션 전역 설정 (싱글톤 캐시)."""
  return Settings()
