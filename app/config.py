"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.3

    # ChromaDB
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "research_docs"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    memory_ttl_seconds: int = 86400

    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_upload_size_mb: int = 50

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
