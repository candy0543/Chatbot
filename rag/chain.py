from __future__ import annotations

import os
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, AzureChatOpenAI


try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore

from app.core.config import Settings
from app.rag.retriever import get_retriever


SYSTEM_PROMPT = (
    "You are an enterprise assistant. Answer strictly using the provided context. "
    "If the answer cannot be found in the context, say you don't know and suggest where to look."
)


def get_llm(cfg: Settings):
    provider = (cfg.llm_provider or "").lower()

    if provider == "openai":
        return ChatOpenAI(model=cfg.llm_model, api_key=cfg.openai_api_key, temperature=0)

    if provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=cfg.azure_openai_deployment,
            api_version=cfg.azure_openai_api_version,
            azure_endpoint=cfg.azure_openai_endpoint,
            api_key=cfg.azure_openai_api_key,
            temperature=0,
        )



    if provider in {"google_genai", "gemini"}:
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError(
                "ChatGoogleGenerativeAI not available. Install 'langchain-google-genai'."
            )
        project = cfg.google_project or cfg.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError(
                "ChatGoogleGenerativeAI requires GOOGLE_PROJECT / PROJECT_ID or GOOGLE_CLOUD_PROJECT env."
            )
        location = cfg.google_location or "us-central1"
        temperature = cfg.model_kwargs.get("temperature", 0)
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            project=project,
            location=location,
            temperature=temperature,
        )

    # Default to OpenAI-compatible error; extend as needed for other providers
    raise ValueError(f"Unsupported llm_provider: {cfg.llm_provider}")

import vertexai
from vertexai.generative_models import GenerativeModel, Tool, grounding


def example_vertex_ai_research() -> None:
    cfg = Settings()
    project = cfg.project_id or cfg.google_project
    if not project:
        raise ValueError("PROJECT_ID or GOOGLE_PROJECT is not configured in settings/.env.")
    if not cfg.data_store_id:
        raise ValueError("DATA_STORE_ID is not configured in settings/.env.")

    vertexai.init(project=project, location=cfg.google_location)

    # Vertex AI Search를 도구로 설정
    tools = [
        Tool.from_retrieval(
            retrieval=grounding.Retrieval(
                source=grounding.VertexAISearch(
                    datastore=cfg.data_store_id,
                    project=project,
                    location=cfg.vertex_search_location,
                )
            )
        )
    ]
    model = GenerativeModel(cfg.vertex_chat_model)
    response = model.generate_content(
        "GCS에 올린 마크다운 문서를 참고해서 사용자의 질문에 답해줘.",
        tools=tools
    )

    print(response.text)
    
def build_rag_chain(cfg: Settings):
    retriever = get_retriever(cfg)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT + "\n\nContext:\n{context}"),
            ("human", "{question}"),
        ]
    )

    llm = get_llm(cfg)

    chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


if __name__ == "__main__":
    example_vertex_ai_research()