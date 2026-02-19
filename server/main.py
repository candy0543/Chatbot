from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.rag.chain import build_rag_chain
from app.rag.llamaindex_engine import build_llamaindex_query_engine

app = FastAPI(title=settings.project_name)

_vertex_chain, _vertex_retriever = build_rag_chain(settings)
_llamaindex_engine = None


class ChatRequest(BaseModel):
    question: str
    engine: str | None = None  # 'vertex' (default) | 'llamaindex'


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    engine = (req.engine or "vertex").lower()
    if engine == "vertex":
        answer = await _vertex_chain.ainvoke(req.question)
        return {"engine_used": "vertex", "answer": answer}
    elif engine == "llamaindex":
        global _llamaindex_engine
        if _llamaindex_engine is None:
            try:
                _llamaindex_engine = build_llamaindex_query_engine(settings)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"LlamaIndex not available: {e}")
        # LlamaIndex engines are sync; run in thread if server is async
        try:
            from anyio.to_thread import run_sync
        except Exception:
            # Fallback: blocking call
            response = _llamaindex_engine.query(req.question)
        else:
            response = await run_sync(_llamaindex_engine.query, req.question)

        # Extract text and sources if available
        text = getattr(response, "response", None) or str(response)
        sources = []
        try:
            for sn in getattr(response, "source_nodes", [])[: settings.top_k]:
                meta = sn.node.metadata or {}
                sources.append({"source": meta.get("source") or meta.get("file_path") or "unknown"})
        except Exception:
            pass
        return {"engine_used": "llamaindex", "answer": text, "sources": sources}
    else:
        raise HTTPException(status_code=400, detail="Invalid engine. Use 'vertex' or 'llamaindex'.")
