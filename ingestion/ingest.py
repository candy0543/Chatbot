from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma

from app.core.config import settings
from app.rag.retriever import get_embeddings


def load_documents(src_dir: Path) -> List:
    docs = []
    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())
        elif suffix in {".docx"}:
            loader = Docx2txtLoader(str(path))
            docs.extend(loader.load())
    return docs


def main() -> None:
    base = settings.base_dir
    src_dir = (base / settings.source_dir).resolve()
    index_dir = (base / settings.vectorstore_dir).resolve()

    src_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    raw_docs = load_documents(src_dir)
    if not raw_docs:
        raise SystemExit(f"No documents found under {src_dir}. Place files and retry.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    splits = splitter.split_documents(raw_docs)

    embeddings = get_embeddings(settings)

    Chroma.from_documents(
        splits,
        embedding=embeddings,
        persist_directory=str(index_dir),
    )

    print(f"Indexed {len(splits)} chunks from {len(raw_docs)} docs into {index_dir}")


if __name__ == "__main__":
    main()
