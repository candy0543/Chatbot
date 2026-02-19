from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, Dict

from app.core.config import settings, Settings

try:
    from chromadb import PersistentClient
    from llama_index.core import VectorStoreIndex
    from llama_index.core import Settings as LISettings
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.vertexai import VertexAIEmbedding
    from llama_index.llms.vertexai import VertexAI
except Exception as e:  # pragma: no cover
    PersistentClient = None  # type: ignore
    VectorStoreIndex = None  # type: ignore
    LISettings = None  # type: ignore
    ChromaVectorStore = None  # type: ignore
    VertexAIEmbedding = None  # type: ignore
    VertexAI = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def build_llamaindex_query_engine(cfg: Settings):
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "LlamaIndex dependencies missing. Install llama-index packages listed in requirements.txt"
        ) from _IMPORT_ERROR

    index_dir = (cfg.base_dir / cfg.vectorstore_dir).resolve()

    # LlamaIndex global settings for LLM and embeddings
    embed_model = VertexAIEmbedding(
        model_name=cfg.vertex_embedding_model,
        project=cfg.google_project,
        location=cfg.google_location,
    )
    llm = VertexAI(
        model=cfg.vertex_chat_model,
        project=cfg.google_project,
        location=cfg.google_location,
        temperature=0.0,
    )
    LISettings.embed_model = embed_model
    LISettings.llm = llm

    # Connect to existing Chroma persistent store (created by our ingestion)
    client = PersistentClient(path=str(index_dir))
    # LangChain's default collection is 'langchain' when using Chroma.from_documents
    vector_store = ChromaVectorStore(chroma_client=client, collection_name="langchain")

    index = VectorStoreIndex.from_vector_store(vector_store)
    return index.as_query_engine(similarity_top_k=cfg.top_k)
