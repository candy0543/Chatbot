from __future__ import annotations

from pathlib import Path
import os
from typing import Optional, Dict, Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_name: str = Field(default="Enterprise RAG Chatbot")

    # Paths
    # Repo root (config.py -> app/core -> app -> repo). parents[2] points to repo root.
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    source_dir: Path = Field(default=Path("data/source"))
    vectorstore_dir: Path = Field(default=Path("data/index"))

    # Chunking
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=150)

    # Retrieval
    top_k: int = Field(default=4)

    # Embeddings
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # model configuration.
    llm_provider: str = Field(default="google_genai")  # ollma, gemini 
    llm_model: str = Field(default="gpt-4o-mini")
    model_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.28, "context_window": 200000, "max_tokens": 1024}
    )

    # OpenAI
    openai_api_key: Optional[str] = None

    # Azure OpenAI
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = Field(default="2024-08-01-preview")
    azure_openai_deployment: Optional[str] = None

    # Ollama (local)
    ollama_model: str = Field(default="llama3.1")

    # Google Vertex AI / Model Garden
    google_project: Optional[str] = Field(default=None, validation_alias="GOOGLE_PROJECT")
    project_id: Optional[str] = Field(default="545718278684", validation_alias="PROJECT_ID")
    google_location: str = Field(default="us-central1")
    google_credentials: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS"),
    )  # Path to service account json (optional)
    vertex_chat_model: str = Field(default="gemini-2.5-flash")
    vertex_embedding_model: str = Field(default="text-embedding-004")
    vertex_search_location: str = Field(default="global")
    # Vertex AI Vector Search (formerly Matching Engine)
    use_vertex_vector_search: bool = Field(default=False)
    data_store_id: Optional[str] = Field(default="source-md_1770015166238", validation_alias="DATA_STORE_ID")
    vertex_index_id: Optional[str] = None
    vertex_index_endpoint_id: Optional[str] = None
    # Google Cloud Storage (for Vertex AI Search data sync)
    gcs_bucket_name: Optional[str] = "source_md"
    gcs_bucket_prefix: Optional[str] = None

    # Notion
    notion_api_key: Optional[str] = "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"
    notion_root_page_id: Optional[str] = None
    notion_database_id: Optional[str] = None
    
    

    def ensure_dirs(self) -> None:
        (self.base_dir / self.source_dir).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.vectorstore_dir).mkdir(parents=True, exist_ok=True)


def _resolve_credentials_path(raw_path: str, base_dir: Path) -> Path:
    """Resolve relative credential paths against the repo root."""
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def _configure_google_credentials(cfg: Settings) -> None:
    explicit_path = (cfg.google_credentials or "").strip()
    if explicit_path:
        resolved = _resolve_credentials_path(explicit_path, cfg.base_dir)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Google credentials file not found at '{resolved}'. Update GOOGLE_CREDENTIALS."
            )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved)
        return

    env_value = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_value is not None and not env_value.strip():
        # Empty env vars cause google.auth to fail before falling back to gcloud/ADC.
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)


settings = Settings()
settings.ensure_dirs()
_configure_google_credentials(settings)
