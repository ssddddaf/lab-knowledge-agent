from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LAB_", env_file=".env", extra="ignore")
    data_dir: Path = Path("data")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    chat_model: str = ""
    qdrant_url: str = "http://localhost:6333"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    enable_vision: bool = False
    enable_dense: bool = True
    enable_ocr: bool = False
    allow_text_only: bool = True

    @property
    def db_path(self) -> Path:
        return self.data_dir / "lab.db"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.db"


@lru_cache
def settings() -> Settings:
    value = Settings()
    value.data_dir.mkdir(parents=True, exist_ok=True)
    return value
