import os
from typing import Optional

class Settings:
    PROJECT_NAME: str = "RealEstate-RAG"
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./chroma")
    DATA_PATH: str = os.getenv("DATA_PATH", "./data/properties.csv")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    HISTORY_BACKEND: str = os.getenv("HISTORY_BACKEND", "shelve")
    DATABASE_URL: Optional[str] = os.getenv("DB_URL")

settings = Settings()