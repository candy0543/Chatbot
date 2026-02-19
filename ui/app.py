from __future__ import annotations

import sys
from pathlib import Path
import asyncio
import chainlit as cl

# Ensure project root is on sys.path when Chainlit loads by file path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.rag.chain import build_rag_chain


@cl.on_chat_start
async def on_chat_start():
    chain, retriever = build_rag_chain(settings)
    cl.user_session.set("chain", chain)
    cl.user_session.set("retriever", retriever)
    await cl.Message(content="RAG assistant ready. Ask about your documents.").send()


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    retriever = cl.user_session.get("retriever")

    if chain is None:
        await cl.Message(content="Session not initialized. Reload the app.").send()
        return

    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    res = await chain.ainvoke(message.content, config={"callbacks": [cb]})

    try:
        docs = await asyncio.to_thread(retriever.get_relevant_documents, message.content)
    except Exception:
        docs = []
    elements = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        elements.append(cl.Text(name=f"source-{i}", content=f"{src}"))

    await cl.Message(content=res, elements=elements).send()
