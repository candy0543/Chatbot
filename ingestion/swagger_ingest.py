from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml

from app.core.config import settings


logger = logging.getLogger(__name__)

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}


def load_openapi(url: str) -> Dict[str, Any]:
    """Load an OpenAPI specification from a Swagger/OpenAPI endpoint."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()

    if "json" in content_type:
        return response.json()
    if "yaml" in content_type or "yml" in content_type:
        return yaml.safe_load(response.text)

    try:
        return response.json()
    except json.JSONDecodeError:
        return yaml.safe_load(response.text)


def _resolve_parameter_ref(param: Dict[str, Any], components: Dict[str, Any]) -> Dict[str, Any]:
    ref = param.get("$ref")
    if not ref:
        return param
    ref_name = ref.split("/")[-1]
    resolved = components.get("parameters", {}).get(ref_name)
    return resolved or param


def split_openapi_to_endpoints(openapi: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the OpenAPI document into a list of endpoint dictionaries."""
    endpoints: List[Dict[str, Any]] = []
    paths = openapi.get("paths", {}) or {}
    components = openapi.get("components", {}) or {}

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        path_level_params = path_item.get("parameters", [])

        for method, spec in path_item.items():
            if method.lower() not in HTTP_METHODS:
                continue
            if not isinstance(spec, dict):
                continue

            method_params = spec.get("parameters", [])
            combined_params = []
            for param in [*path_level_params, *method_params]:
                combined_params.append(_resolve_parameter_ref(param, components))

            endpoint = {
                "path": path,
                "method": method.upper(),
                "summary": spec.get("summary") or "",
                "description": spec.get("description") or "",
                "tags": spec.get("tags") or [],
                "operation_id": spec.get("operationId")
                or f"{method}_{path}".replace("/", "_").replace("{", "").replace("}", ""),
                "parameters": combined_params,
                "request_body": spec.get("requestBody") or {},
                "responses": spec.get("responses") or {},
            }
            endpoints.append(endpoint)

    return endpoints


def _format_schema_block_text(payload: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    schema = payload.get("schema")
    if schema:
        dumped = yaml.safe_dump(schema, sort_keys=False, width=120).strip()
        if dumped:
            lines.append("Schema:")
            lines.extend(dumped.splitlines())
    example = payload.get("example")
    examples = payload.get("examples")
    if example:
        if lines:
            lines.append("")
        example_dump = yaml.safe_dump(example, sort_keys=False, width=120).strip()
        lines.append("Example:")
        lines.extend(example_dump.splitlines())
    elif examples:
        if lines:
            lines.append("")
        example_dump = yaml.safe_dump(examples, sort_keys=False, width=120).strip()
        lines.append("Examples:")
        lines.extend(example_dump.splitlines())
    return lines


def _indent_lines(lines: List[str], indent: str = "    ") -> List[str]:
    return [f"{indent}{line}" if line else "" for line in lines]


def openapi_endpoint_to_text(endpoint: Dict[str, Any]) -> str:
    """Convert a single OpenAPI endpoint into a plain-text snippet."""
    lines: List[str] = []
    header = f"{endpoint['method']} {endpoint['path']}"
    lines.append(header)
    lines.append("-" * len(header))

    if endpoint.get("summary"):
        lines.append(f"Summary: {endpoint['summary']}")
    if endpoint.get("description"):
        lines.append(endpoint["description"].strip())
    if endpoint.get("tags"):
        lines.append(f"Tags: {', '.join(endpoint['tags'])}")

    parameters = endpoint.get("parameters") or []
    if parameters:
        lines.append("")
        lines.append("Parameters:")
        for param in parameters:
            name = param.get("name", "?")
            location = param.get("in", "n/a")
            schema = param.get("schema") or {}
            schema_type = schema.get("type") or schema.get("$ref", "object")
            required = "required" if param.get("required") else "optional"
            description = (param.get("description") or "").strip()
            lines.append(
                f"- {name} ({location}, type={schema_type}, {required})"
            )
            if description:
                lines.append(f"  {description}")

    request_body = endpoint.get("request_body") or {}
    if request_body:
        lines.append("")
        lines.append("Request Body:")
        lines.append(f"Required: {'yes' if request_body.get('required') else 'no'}")
        for mime, payload in (request_body.get("content") or {}).items():
            lines.append(f"- {mime}")
            lines.extend(_indent_lines(_format_schema_block_text(payload)))

    responses = endpoint.get("responses") or {}
    if responses:
        lines.append("")
        lines.append("Responses:")
        for code, response in responses.items():
            description = (response.get("description") or "").strip()
            lines.append(f"- {code}: {description}")
            for mime, payload in (response.get("content") or {}).items():
                lines.append(f"  {mime}")
                lines.extend(_indent_lines(_format_schema_block_text(payload), indent="    "))

    return "\n".join(lines).strip() + "\n"
        
def swagger_url_to_text_docs(url: str) -> List[Dict[str, Any]]:
    """Download swagger spec and return per-endpoint plain-text plus metadata."""
    openapi = load_openapi(url)
    endpoints = split_openapi_to_endpoints(openapi)
    docs: List[Dict[str, Any]] = []
    for endpoint in endpoints:
        doc = {**endpoint, "text": openapi_endpoint_to_text(endpoint)}
        docs.append(doc)
    return docs


def endpoint_to_jsonl(endpoint: Dict[str, Any]) -> str:
    payload = {
        "id": endpoint.get("operation_id") or f"{endpoint['method']}_{endpoint['path']}",
        "content": endpoint.get("text") or openapi_endpoint_to_text(endpoint),
        "metadata": {
            "path": endpoint["path"],
            "method": endpoint["method"],
            "tags": endpoint.get("tags", []),
            "source": "swagger_api",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _slugify_endpoint(method: str, path: str) -> str:
    slug = path.strip("/") or "root"
    slug = slug.replace("/", "_")
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", slug)
    return f"{method.lower()}_{slug}"


def save_text_documents(
    docs: List[Dict[str, Any]],
    output_dir: Path | str,
    combined_filename: str | None = None,
) -> List[Path]:
    """Persist plain-text docs to disk, returning the written paths."""
    if not docs:
        raise ValueError("No swagger documents to persist")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_paths: List[Path] = []

    if combined_filename:
        combined_path = output_path / combined_filename
        with combined_path.open("w", encoding="utf-8") as handle:
            for doc in docs:
                handle.write(doc["text"])
                handle.write("\n")
        written_paths.append(combined_path)
        return written_paths

    for doc in docs:
        file_name = f"{_slugify_endpoint(doc['method'], doc['path'])}.txt"
        doc_path = output_path / file_name
        with doc_path.open("w", encoding="utf-8") as handle:
            handle.write(doc["text"])
        written_paths.append(doc_path)

    return written_paths


def default_output_dir() -> Path:
    return (settings.base_dir / settings.source_dir / "swagger").resolve()


def fetch_and_save_swagger_text(
    url: str, output_dir: Path | str | None = None, combined_filename: str | None = None
) -> List[Path]:
    docs = swagger_url_to_text_docs(url)
    target_dir = output_dir or default_output_dir()
    return save_text_documents(docs, target_dir, combined_filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Swagger docs and save as plain text.")
    parser.add_argument("--url", default="https://server.cinelab.co.kr/v3/api-docs" , help="Swagger/OpenAPI endpoint URL")
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory to store generated text files",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Store all endpoints in a single text file",
    )
    parser.add_argument(
        "--filename",
        default="swagger_endpoints.txt",
        help="Filename when using --combined",
    )
    args = parser.parse_args()

    combined_filename = args.filename if args.combined else None

    try:
        written = fetch_and_save_swagger_text(
            args.url, args.output_dir, combined_filename=combined_filename
        )
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        logger.error("Failed to save swagger docs: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Saved %s text files to %s", len(written), args.output_dir)


if __name__ == "__main__":
    main()