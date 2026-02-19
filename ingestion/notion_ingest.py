from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple

from notion_client import Client as NotionClient
from notion_client.helpers import iterate_paginated_api

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import settings
from app.rag.retriever import get_embeddings, get_vectorstore


INVALID_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
DEFAULT_PUBLISHER = "Notion"
NOTION_TAG = "notion"


@dataclass
class NotionPage:
    id: str
    title: str
    url: str
    last_edited_time: Optional[str]
    properties: Dict
    root_id: str
    root_title: Optional[str]


def normalize_notion_id(value: str) -> str:
    """Extract and canonicalize a Notion page/database id from a url or raw id."""

    if not value:
        raise ValueError("A Notion identifier or URL is required.")

    cleaned = value.strip().split("?")[0].split("#")[0]
    hex_chars = re.sub(r"[^0-9a-fA-F]", "", cleaned)
    if len(hex_chars) < 32:
        raise ValueError(f"Unable to parse Notion id from '{value}'.")
    token = hex_chars[-32:].lower()
    return f"{token[0:8]}-{token[8:12]}-{token[12:16]}-{token[16:20]}-{token[20:32]}"


def sanitize_title_for_path(title: str, fallback: str) -> str:
    candidate = title.strip() if title else ""
    candidate = INVALID_FILENAME_CHARS.sub("_", candidate)
    candidate = candidate.strip("._- ")
    return candidate or fallback


def _unique_stem(preferred: str, taken: Set[str]) -> str:
    if preferred not in taken:
        taken.add(preferred)
        return preferred

    idx = 2
    while True:
        candidate = f"{preferred}-{idx}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
        idx += 1


def _normalize_id_list(values: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for raw in values:
        if not raw:
            continue
        canonical = normalize_notion_id(raw)
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _plain_text(rich_text_array: List[Dict]) -> str:
    return "".join([rt.get("plain_text", "") for rt in rich_text_array or []])


def _get_page_title(page: Dict) -> str:
    props = page.get("properties", {})
    for name, val in props.items():
        if isinstance(val, dict) and val.get("type") == "title":
            title_arr = val.get("title", [])
            txt = _plain_text(title_arr).strip()
            if txt:
                return txt
    # Fallbacks
    return page.get("url") or page.get("id")


def _derive_publisher(simple_props: Dict[str, Any]) -> Optional[str]:
    for key, value in (simple_props or {}).items():
        if not value:
            continue
        lowered = key.lower()
        if "publisher" in lowered or lowered in {"owner", "author"}:
            return str(value)
    return None


def _extract_block_text(block: Dict) -> str:
    t = block.get("type")
    data = block.get(t, {}) if t else {}
    if not isinstance(data, dict):
        return ""

    text = _plain_text(data.get("rich_text", []))
    if t == "heading_1":
        return f"# {text}" if text else ""
    if t == "heading_2":
        return f"## {text}" if text else ""
    if t == "heading_3":
        return f"### {text}" if text else ""
    if t == "bulleted_list_item":
        return f"- {text}" if text else ""
    if t == "numbered_list_item":
        return f"1. {text}" if text else ""
    if t == "to_do":
        checkbox = "[x]" if data.get("checked") else "[ ]"
        return f"- {checkbox} {text}" if text else ""
    if t in {"toggle", "paragraph", "synced_block"}:
        return text
    if t in {"quote", "callout"}:
        return f"> {text}" if text else ""
    if t == "code":
        language = data.get("language") or ""
        code_text = text or ""
        fence = f"```{language}\n{code_text}\n```" if code_text else ""
        return fence

    if t == "equation":
        expr = data.get("expression", "")
        return f"$ {expr} $" if expr else ""

    if t == "table_row":
        cells = data.get("cells", [])
        cell_text = [" | ".join(_plain_text(cell) for cell in row) for row in cells]
        return "\n".join(cell_text)

    # For other types (divider, image, file, etc.) return empty or a placeholder.
    return ""


def _list_children(client: NotionClient, block_id: str) -> List[Dict]:
    results: List[Dict] = []
    for page in iterate_paginated_api(client.blocks.children.list, block_id=block_id):
        results.extend(page.get("results", []))
    return results


def _collect_child_links(client: NotionClient, page_id: str) -> Tuple[List[str], List[str]]:
    child_pages: List[str] = []
    child_dbs: List[str] = []
    blocks = _list_children(client, page_id)
    for b in blocks:
        btype = b.get("type")
        if btype == "child_page":
            child_pages.append(b.get("id"))
        elif btype == "child_database":
            child_dbs.append(b.get("id"))
    return child_pages, child_dbs


def _get_page_text_recursive(client: NotionClient, page_id: str) -> str:
    lines: List[str] = []

    def walk(block_id: str, depth: int = 0) -> None:
        blocks = _list_children(client, block_id)
        for b in blocks:
            txt = _extract_block_text(b)
            if txt:
                lines.append(txt)
            if b.get("has_children"):
                walk(b.get("id"), depth + 1)

    walk(page_id)
    text = "\n".join(lines)
    # Normalize whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _iter_database_pages(client: NotionClient, database_id: str) -> Generator[Dict, None, None]:
    for page in iterate_paginated_api(client.databases.query, database_id=database_id):
        for res in page.get("results", []):
            yield res


def iter_pages(
    client: NotionClient,
    root_page_ids: Optional[Sequence[str]] = None,
    database_ids: Optional[Sequence[str]] = None,
) -> Generator[NotionPage, None, None]:
    visited: Set[str] = set()
    stack: List[Tuple[str, str, Optional[str], Optional[str]]] = []  # (kind, id, root_id, root_title)

    if database_ids:
        for dbid in reversed([db for db in database_ids if db]):
            stack.append(("db", dbid, dbid, None))
    if root_page_ids:
        for pid in reversed([pid for pid in root_page_ids if pid]):
            stack.append(("page", pid, pid, None))

    while stack:
        kind, rid, root_id, root_title = stack.pop()
        if kind == "db":
            if rid in visited:
                continue
            visited.add(rid)
            for page in _iter_database_pages(client, rid):
                pid = page.get("id")
                if not pid:
                    continue
                title = _get_page_title(page)
                page_root_id = root_id or pid
                page_root_title = root_title or title
                yield NotionPage(
                    id=pid,
                    title=title,
                    url=page.get("url", ""),
                    last_edited_time=page.get("last_edited_time"),
                    properties=page.get("properties", {}),
                    root_id=page_root_id,
                    root_title=page_root_title,
                )
        else:  # page
            if rid in visited:
                continue
            visited.add(rid)
            page = client.pages.retrieve(page_id=rid)
            title = _get_page_title(page)
            current_root_id = root_id or rid
            current_root_title = root_title or title
            yield NotionPage(
                id=rid,
                title=title,
                url=page.get("url", ""),
                last_edited_time=page.get("last_edited_time"),
                properties=page.get("properties", {}),
                root_id=current_root_id,
                root_title=current_root_title,
            )
            # Enqueue children
            child_pages, child_dbs = _collect_child_links(client, rid)
            for cp in child_pages:
                stack.append(("page", cp, current_root_id, current_root_title))
            for db in child_dbs:
                stack.append(("db", db, current_root_id, current_root_title))


def build_documents(
    client: NotionClient,
    root_page_ids: Optional[Sequence[str]] = None,
    database_ids: Optional[Sequence[str]] = None,
) -> List[Document]:
    docs: List[Document] = []
    for page in iter_pages(client, root_page_ids=root_page_ids, database_ids=database_ids):
        content = _get_page_text_recursive(client, page.id)
        if not content:
            continue
        metadata: Dict[str, Any] = {
            "notion_id": page.id,
            "title": page.title,
            "url": page.url,
            "last_edited_time": page.last_edited_time,
            "source": NOTION_TAG,
        }
        simple_props: Dict[str, Any] = {}
        for k, v in (page.properties or {}).items():
            if isinstance(v, dict) and v.get("type") == "rich_text":
                simple_props[k] = _plain_text(v.get("rich_text", []))
            elif isinstance(v, dict) and v.get("type") == "title":
                simple_props[k] = _plain_text(v.get("title", []))
            elif isinstance(v, dict) and v.get("type") in {"number", "checkbox", "url", "email", "phone_number"}:
                simple_props[k] = v.get(v.get("type"))
            elif isinstance(v, dict) and v.get("type") in {"select"}:
                opt = v.get("select")
                if isinstance(opt, dict):
                    simple_props[k] = opt.get("name")
        if simple_props:
            metadata["properties"] = simple_props

        metadata["publisher"] = _derive_publisher(simple_props) or DEFAULT_PUBLISHER
        metadata["tags"] = [NOTION_TAG]
        metadata["root_id"] = page.root_id
        metadata["root_title"] = page.root_title or page.title

        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def _format_scalar_for_yaml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if value is None:
        return '""'
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _yaml_dump_lines(value: Any, indent: int = 0) -> List[str]:
    prefix = " " * indent
    lines: List[str] = []
    if isinstance(value, dict):
        if not value:
            lines.append(f"{prefix}{{}}")
            return lines
        for key, val in value.items():
            if isinstance(val, dict):
                if not val:
                    lines.append(f"{prefix}{key}: {{}}")
                else:
                    lines.append(f"{prefix}{key}:")
                    lines.extend(_yaml_dump_lines(val, indent + 2))
            elif isinstance(val, list):
                if not val:
                    lines.append(f"{prefix}{key}: []")
                else:
                    lines.append(f"{prefix}{key}:")
                    lines.extend(_yaml_dump_lines(val, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar_for_yaml(val)}")
    elif isinstance(value, list):
        if not value:
            lines.append(f"{prefix}[]")
        else:
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.extend(_yaml_dump_lines(item, indent + 2))
                else:
                    lines.append(f"{prefix}- {_format_scalar_for_yaml(item)}")
    else:
        lines.append(f"{prefix}{_format_scalar_for_yaml(value)}")
    return lines


def build_front_matter(metadata: Dict[str, Any]) -> str:
    fm = OrderedDict()
    fm["title"] = metadata.get("title") or "Untitled Notion Page"
    fm["url"] = metadata.get("url") or ""
    tags = metadata.get("tags") or []
    fm["tags"] = sorted(set(tags + [NOTION_TAG])) if tags else [NOTION_TAG]
    fm["last_modified"] = metadata.get("last_edited_time") or ""
    fm["publisher"] = metadata.get("publisher") or DEFAULT_PUBLISHER
    fm["notion_id"] = metadata.get("notion_id") or ""
    properties = metadata.get("properties") or {}
    if properties:
        sorted_props = OrderedDict(sorted(properties.items(), key=lambda item: item[0].lower()))
        fm["properties"] = sorted_props

    lines = ["---"]
    lines.extend(_yaml_dump_lines(fm))
    lines.append("---")
    return "\n".join(lines)


def _write_markdown_files(documents: Sequence[Document], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    taken: Dict[Path, Set[str]] = {}
    written: List[Path] = []

    for doc in documents:
        metadata = doc.metadata or {}
        root_title = metadata.get("root_title") or metadata.get("title") or metadata.get("url")
        root_fallback = f"notion-root-{(metadata.get('root_id') or metadata.get('notion_id') or '')[:8]}".strip("-") or "notion-root"
        root_slug = sanitize_title_for_path(root_title or "", root_fallback)
        root_dir = output_dir / root_slug
        root_dir.mkdir(parents=True, exist_ok=True)

        title = metadata.get("title") or metadata.get("url") or metadata.get("notion_id") or "notion-page"
        fallback = f"notion-{(metadata.get('notion_id') or '')[:8]}".strip("-") or "notion-page"
        slug = sanitize_title_for_path(title, fallback)
        stem_pool = taken.setdefault(root_dir, set())
        unique_stem = _unique_stem(slug, stem_pool)
        file_path = root_dir / f"{unique_stem}.md"

        content_body = doc.page_content.strip()
        if not content_body:
            continue

        front_matter = build_front_matter(metadata)
        file_path.write_text(f"{front_matter}\n\n{content_body}\n", encoding="utf-8")
        written.append(file_path)

    return written


def _fetch_notion_documents(page_list: Sequence[str]) -> List[Document]:
    if not settings.notion_api_key:
        raise SystemExit("NOTION_API_KEY (settings.notion_api_key) is required")

    normalized_pages: List[str] = []
    if page_list:
        normalized_pages = _normalize_id_list(page_list)
    elif settings.notion_root_page_id:
        normalized_pages = _normalize_id_list([settings.notion_root_page_id])

    database_ids: List[str] = []
    if settings.notion_database_id:
        database_ids = _normalize_id_list([settings.notion_database_id])

    if not normalized_pages and not database_ids:
        raise SystemExit("Provide at least one Notion page or database id to export.")

    client = NotionClient(auth=settings.notion_api_key)
    documents = build_documents(
        client,
        root_page_ids=normalized_pages or None,
        database_ids=database_ids or None,
    )
    if not documents:
        raise SystemExit("No Notion content found to export.")

    return documents


def export_notion_pages_to_markdown(page_list: Sequence[str], output_subdir: str = "notion") -> List[Path]:
    documents = _fetch_notion_documents(page_list)

    output_dir = (settings.base_dir / settings.source_dir / output_subdir).resolve()
    written = _write_markdown_files(documents, output_dir)
    if not written:
        raise SystemExit("Notion pages contained no textual content to export.")

    print(f"Exported {len(written)} markdown files to {output_dir}")
    return written

#vertex ai search 안쓰면. 
def ingest_from_notion(page_list: List[str]) -> None:
    raw_docs = _fetch_notion_documents(page_list)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    splits = splitter.split_documents(raw_docs)

    embeddings = get_embeddings(settings)
    if embeddings is None:
        raise SystemExit("Embeddings provider not configured. Set llm_provider=vertex or google.")

    vectorstore = get_vectorstore(settings)
    # Add documents to vector store
    vectorstore.add_documents(splits)

    # Persist for local stores like Chroma
    persist = getattr(vectorstore, "persist", None)
    if callable(persist):
        persist()

    print(f"Indexed {len(splits)} chunks from {len(raw_docs)} Notion pages")

def main():
    
    page_list =[
        #"https://www.notion.so/hi-stranger/About-HiStranger-52cbaf38773c45f7b780ce0a26fccf67",
        #"https://www.notion.so/hi-stranger/Wiki-15a491a72d5b807391a6d707b31e3128",
        "https://www.notion.so/hi-stranger/GCP-DB-SQL-28d491a72d5b80f1a99feba6b1a4149a",
        #"https://www.notion.so/hi-stranger/25-07-18_-_-232491a72d5b808db04dd4bd6bd7f93e",
        "https://www.notion.so/hi-stranger/26-01-09-2e2491a72d5b80a1ab66ef8337c5fffb",
        "https://www.notion.so/hi-stranger/298491a72d5b80da8ec9e0d17f61524c"
    ]
    
    export_notion_pages_to_markdown(page_list)
    
    
if __name__ == "__main__":
    main()
