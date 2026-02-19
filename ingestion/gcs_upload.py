from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

from google.api_core.exceptions import NotFound
from google.cloud import storage

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class UploadReport:

    uploaded: list[str]
    skipped: list[str]
    deleted: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return {"uploaded": self.uploaded, "skipped": self.skipped, "deleted": self.deleted}


def _normalize_prefix(prefix: str | None) -> str:
    """Return a safe blob prefix that always ends with '/' or empty string."""
    if not prefix:
        return ""
    cleaned = prefix.strip("/")
    return f"{cleaned}/" if cleaned else ""


def _build_blob_name(prefix: str, relative_path: Path) -> str:
    """Compose the blob path using the normalized prefix and relative file path."""
    rel_key = relative_path.as_posix()
    return f"{prefix}{rel_key}" if prefix else rel_key


def _iter_local_files(root: Path) -> Iterator[Tuple[Path, Path]]:
    for candidate in root.rglob("*"):
        if candidate.is_file():
            yield candidate, candidate.relative_to(root)


def _should_upload(blob: storage.Blob, file_size: int, force: bool) -> bool:
    if force:
        return True
    try:
        blob.reload()
    except NotFound:
        return True
    return blob.size != file_size


def upload_data_source(
    bucket_name: str,
    source_dir: Path,
    prefix: str | None = None,
    *,
    dry_run: bool = False,
    force: bool = False,
    delete_extra: bool = False,
) -> UploadReport:
    """Sync the local data/source directory to a GCS bucket for Vertex AI Search."""

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    normalized_prefix = _normalize_prefix(prefix)

    client_kwargs: dict[str, str] = {}
    if settings.google_project:
        client_kwargs["project"] = settings.google_project
    client = storage.Client(**client_kwargs)

    bucket = client.bucket(bucket_name)
    if not bucket.exists():  
        raise ValueError(f"Bucket {bucket_name} does not exist or you lack permissions.")

    uploaded: list[str] = []
    skipped: list[str] = []
    deleted: list[str] = []
    keep_names: set[str] = set()
    file_count = 0

    for file_path, rel_path in _iter_local_files(source_dir):
        file_count += 1
        blob_name = _build_blob_name(normalized_prefix, rel_path)
        keep_names.add(blob_name)
        blob = bucket.blob(blob_name)
        file_size = file_path.stat().st_size

        if not _should_upload(blob, file_size, force):
            skipped.append(blob_name)
            continue

        if dry_run:
            logger.info("[DRY RUN] upload %s -> gs://%s/%s", file_path, bucket_name, blob_name)
        else:
            blob.upload_from_filename(str(file_path))
            logger.info("Uploaded %s -> gs://%s/%s", file_path, bucket_name, blob_name)
        uploaded.append(blob_name)

    if file_count == 0:
        raise ValueError(f"No files found under {source_dir} to upload.")

    if delete_extra:
        list_prefix = normalized_prefix or None
        for blob in bucket.list_blobs(prefix=list_prefix):
            if blob.name in keep_names:
                continue
            if dry_run:
                logger.info("[DRY RUN] delete gs://%s/%s", bucket_name, blob.name)
            else:
                blob.delete()
                logger.info("Deleted gs://%s/%s", bucket_name, blob.name)
            deleted.append(blob.name)

    return UploadReport(uploaded=uploaded, skipped=skipped, deleted=deleted)


def _default_source_dir() -> Path:
    return (settings.base_dir / settings.source_dir).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload local documents to Google Cloud Storage for Vertex AI Search ingestion."
    )
    parser.add_argument(
        "--bucket",
        default=settings.gcs_bucket_name,
        help="Target GCS bucket name. Defaults to GCS_BUCKET_NAME env var.",
    )
    parser.add_argument(
        "--prefix",
        default=settings.gcs_bucket_prefix,
        help="Optional folder/prefix inside the bucket (e.g. 'vertex/docs').",
    )
    parser.add_argument(
        "--source-dir",
        default=str(_default_source_dir()),
        help="Directory to upload (defaults to data/source).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without uploading")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload every file even if size matches the remote copy.",
    )
    parser.add_argument(
        "--delete-extra",
        action="store_true",
        help="Delete remote blobs that are not present locally within the prefix.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    if not args.bucket:
        raise SystemExit("--bucket is required (or set GCS_BUCKET_NAME in your .env)")

    report = upload_data_source(
        bucket_name=args.bucket,
        source_dir=Path(args.source_dir),
        prefix=args.prefix,
        dry_run=args.dry_run,
        force=args.force,
        delete_extra=args.delete_extra,
    )
    logger.info(
        "Sync complete: %s uploaded, %s skipped, %s deleted",
        len(report.uploaded),
        len(report.skipped),
        len(report.deleted),
    )


if __name__ == "__main__":
    main()
