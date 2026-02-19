from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from pydantic import PrivateAttr
import logging
from functools import lru_cache

try:
    from langchain_google_vertexai import VertexAIEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    VertexAIEmbeddings = None  # type: ignore

try:
    from google.cloud import discoveryengine_v1  # type: ignore
except Exception:  # pragma: no cover
    discoveryengine_v1 = None  # type: ignore

try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # type: ignore

try:
    from google.protobuf.json_format import MessageToDict  # type: ignore
except Exception:  # pragma: no cover
    MessageToDict = None  # type: ignore

from app.core.config import Settings


def _normalize_payload(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_payload(item) for item in obj]
    if MessageToDict is not None and hasattr(obj, "_pb"):
        try:
            return _normalize_payload(MessageToDict(obj._pb))
        except AttributeError:
            pass
    if hasattr(obj, "items"):
        try:
            return {k: _normalize_payload(v) for k, v in obj.items()}  # type: ignore[attr-defined]
        except Exception:
            pass
    return obj


def _struct_to_dict(struct_obj: Any) -> Dict[str, Any]:
    if not struct_obj:
        return {}
    if MessageToDict is not None and hasattr(struct_obj, "_pb"):
        try:
            return _normalize_payload(MessageToDict(struct_obj._pb))  # type: ignore[return-value]
        except AttributeError:
            # Map containers exposed by the protobuf runtime do not implement DESCRIPTOR.
            # Fall back to a manual conversion path below.
            pass
    try:
        return _normalize_payload(dict(struct_obj))
    except TypeError:  # pragma: no cover
        return {}


def _extract_text(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    logging.info("Extracting text from payload: %s", payload)

    extractive_segments = payload.get("extractive_segments")
    logging.info("Extractive segments found: %s", extractive_segments)
    if isinstance(extractive_segments, list) and extractive_segments:
        first_segment = extractive_segments[0]
        if isinstance(first_segment, dict):
            seg_text = first_segment.get("content") or first_segment.get("text")
            if isinstance(seg_text, str) and seg_text.strip():
                return seg_text.strip()

    def _from_segments(segments: Any) -> List[str]:
        texts: List[str] = []
        if isinstance(segments, list):
            for segment in segments:
                if isinstance(segment, dict):
                    seg_text = segment.get("content") or segment.get("text")
                    if seg_text:
                        texts.append(str(seg_text).strip())
        return texts

    candidates: List[str] = []
    for key in ("extractive_segments", "extractive_answers", "chunks"):
        candidates.extend(_from_segments(payload.get(key)))

    for key in ("chunk", "content", "text", "snippet", "title"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    deduped = [text for i, text in enumerate(candidates) if text and text not in candidates[:i]]
    return "\n".join(deduped)


@lru_cache(maxsize=64)
def _read_gcs_blob(uri: str) -> str:
    if storage is None or not uri.startswith("gs://"):
        return ""
    _, remainder = uri.split("gs://", 1)
    if "/" not in remainder:
        return ""
    bucket_name, blob_name = remainder.split("/", 1)
    if not bucket_name or not blob_name:
        return ""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            return ""
        return blob.download_as_text()
    except Exception:
        return ""


class VertexAISearchRetriever(BaseRetriever):
    project_id: str
    data_store_id: str
    location: str = "global"
    max_documents: int = 4

    _client: Any = PrivateAttr(default=None)
    _serving_config: str = PrivateAttr(default="")

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        if discoveryengine_v1 is None:
            raise RuntimeError(
                "google-cloud-discoveryengine is required for Vertex AI Search retrieval."
            )
        location = self.location or "global"
        endpoint = f"{location}-discoveryengine.googleapis.com"
        self._client = discoveryengine_v1.SearchServiceClient(
            client_options={"api_endpoint": endpoint}
        )
        self._serving_config = (
            f"projects/{self.project_id}/locations/{location}/collections/default_collection/"
            f"dataStores/{self.data_store_id}/servingConfigs/default_serving_config"
        )

    def _search(self, query: str) -> List[Document]:
        request = discoveryengine_v1.SearchRequest(
            serving_config=self._serving_config,
            query=query,
            page_size=self.max_documents,
            query_expansion_spec=discoveryengine_v1.SearchRequest.QueryExpansionSpec(
                condition=discoveryengine_v1.SearchRequest.QueryExpansionSpec.Condition.AUTO
            ),
            spell_correction_spec=discoveryengine_v1.SearchRequest.SpellCorrectionSpec(
                mode=discoveryengine_v1.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            ),
            content_search_spec=discoveryengine_v1.SearchRequest.ContentSearchSpec(
                snippet_spec=discoveryengine_v1.SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True
                ),
                extractive_content_spec=discoveryengine_v1.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    #max_extractive_answer_count=1,    
                    max_extractive_segment_count=3, 
                    return_extractive_segment_score=True
                ),
            ),
           
        )
        response = self._client.search(request=request)
        docs: List[Document] = []
        for result in response:
            derived = _struct_to_dict(result.document.derived_struct_data)
            struct_data = _struct_to_dict(result.document.struct_data)
            logging.info("Vertex AI Search retrieved derived_data: %s", derived.get("extractive_segments")[0].get("content") if derived.get("extractive_segments") else "No extractive_segments")
            logging.info("Vertex AI Search retrieved struct_data: %s", struct_data)

            page_content =  derived.get("extractive_segments")[0].get("content") if derived.get("extractive_segments") else "No extractive_segments" #_extract_text(derived) or _extract_text(struct_data)
            logging.info("Vertex AI Search retrieved page_content: %s", page_content)
            if not page_content:
                page_content = "No matching content returned for this chunk."

            metadata: Dict[str, Any] = {
                "vertex_document_id": result.document.id,
            }
            source = (
                derived.get("link")
                or derived.get("uri")
                or struct_data.get("link")
                or struct_data.get("uri")
                or result.document.id
            )
            metadata["source"] = source
            metadata.update(
                {
                    key: value
                    for key, value in struct_data.items()
                    if key not in metadata and isinstance(key, str)
                }
            )
            if (not page_content or page_content == "No matching content returned for this chunk.") and source:
                external_text = _read_gcs_blob(source)
                if external_text:
                    page_content = external_text
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> List[Document]:  # type: ignore[override]
        return self._search(query)

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any) -> List[Document]:  # type: ignore[override]
        return await asyncio.to_thread(self._search, query)


def get_embeddings(cfg: Settings):
    provider = cfg.llm_provider.lower()
    if provider == "openai":
        return OpenAIEmbeddings(api_key=cfg.openai_api_key)
    if provider == "azure":
        return AzureOpenAIEmbeddings(
            api_key=cfg.azure_openai_api_key,
            azure_endpoint=cfg.azure_openai_endpoint,
            api_version=cfg.azure_openai_api_version,
            azure_deployment=cfg.azure_openai_deployment,
        )
    if provider in {"vertex", "google"}:
        if VertexAIEmbeddings is None:
            raise RuntimeError(
                "Vertex AI Embeddings not available. Install 'langchain-google-vertexai'."
            )
        project = cfg.google_project or cfg.project_id
        if not project:
            raise ValueError("Vertex AI Embeddings require PROJECT_ID or GOOGLE_PROJECT.")
        return VertexAIEmbeddings(
            model_name=cfg.vertex_embedding_model,
            project=project,
            location=cfg.google_location,
        )
    return None


def get_vectorstore(cfg: Settings) -> Chroma:
    index_dir = (cfg.base_dir / cfg.vectorstore_dir).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings(cfg)
    if (
        getattr(cfg, "use_vertex_vector_search", False)
        and cfg.vertex_index_id
        and cfg.vertex_index_endpoint_id
    ):
        try:
            from langchain_google_vertexai.vectorstores import MatchingEngine  # type: ignore

            project = cfg.google_project or cfg.project_id
            if not project:
                raise ValueError(
                    "Vertex AI Vector Search requires PROJECT_ID or GOOGLE_PROJECT to be configured."
                )
            return MatchingEngine.from_existing_index(  # type: ignore
                embedding=embeddings,
                project_id=project,
                region=cfg.google_location,
                index_id=cfg.vertex_index_id,
                endpoint_id=cfg.vertex_index_endpoint_id,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to initialize Vertex AI Vector Search. Ensure 'langchain-google-vertexai' is installed "
                "and index/endpoint IDs are correct."
            ) from e

    return Chroma(embedding_function=embeddings, persist_directory=str(index_dir))


def get_retriever(cfg: Settings):
    if cfg.data_store_id:
        logging.info("Using Vertex AI Search Retriever as data_store_id is configured.")
        project = cfg.project_id or cfg.google_project
        if not project:
            raise ValueError(
                "Vertex AI Search requires PROJECT_ID or GOOGLE_PROJECT to be configured."
            )
        location = cfg.vertex_search_location or "global"
        return VertexAISearchRetriever(
            project_id=project,
            data_store_id=cfg.data_store_id,
            location=location,
            max_documents=cfg.top_k,
        )

    vectorstore = get_vectorstore(cfg)
    return vectorstore.as_retriever(search_kwargs={"k": cfg.top_k})
