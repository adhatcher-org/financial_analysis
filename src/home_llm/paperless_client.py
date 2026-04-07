from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import requests

from home_llm.config import PaperlessConfig
from home_llm.models import DocumentChunk


@dataclass(slots=True)
class PaperlessDocument:
    document_id: int
    modified_at: float
    chunks: list[DocumentChunk]
    metadata: dict[str, Any]


class PaperlessClient:
    def __init__(self, config: PaperlessConfig) -> None:
        if not config.base_url:
            raise ValueError("Paperless is enabled but [paperless].base_url is missing.")
        if not config.token:
            raise ValueError("Paperless is enabled but [paperless].token is missing.")
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {config.token}",
                "Accept": "application/json",
            }
        )

    def iter_documents(self) -> list[PaperlessDocument]:
        documents: list[PaperlessDocument] = []
        next_url = f"{self.config.base_url}/api/documents/?page_size={self.config.page_size}"

        while next_url:
            response = self.session.get(next_url, timeout=60)
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("results", []):
                doc = self._to_document(item)
                if doc is not None:
                    documents.append(doc)
            next_url = payload.get("next")

        return documents

    def _to_document(self, item: dict[str, Any]) -> PaperlessDocument | None:
        content = str(item.get("content") or "").strip()
        if not content:
            return None

        document_id = int(item["id"])
        title = str(item.get("title") or f"paperless-{document_id}")
        archive_name = str(item.get("archive_serial_number") or "").strip()
        filename = archive_name or f"{title}.paperless.txt"
        storage_path = str(item.get("storage_path") or "paperless")
        file_path = str(PurePosixPath("/paperless", storage_path, filename))
        doc_type = _detect_doc_type(item)
        modified_at = _parse_modified_at(item)

        chunk = DocumentChunk(
            file_path=file_path,
            doc_type=doc_type,
            page_label="paperless-content",
            content=content,
        )
        return PaperlessDocument(
            document_id=document_id,
            modified_at=modified_at,
            chunks=[chunk],
            metadata={
                "paperless_id": document_id,
                "title": title,
                "created": item.get("created"),
                "added": item.get("added"),
                "storage_path": item.get("storage_path"),
                "document_type": item.get("document_type"),
                "correspondent": item.get("correspondent"),
                "tags": item.get("tags", []),
            },
        )


def _detect_doc_type(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "").lower()
    storage_path = str(item.get("storage_path") or "").lower()
    text = f"{title} {storage_path}"
    checks = [
        ("mortgage", "mortgage"),
        ("loan", "loan"),
        ("insurance", "insurance"),
        ("brokerage", "investment"),
        ("invest", "investment"),
        ("retirement", "retirement"),
        ("bank", "bank"),
        ("statement", "statement"),
        ("policy", "insurance"),
    ]
    for needle, label in checks:
        if needle in text:
            return label
    return "paperless"


def _parse_modified_at(item: dict[str, Any]) -> float:
    for key in ("modified", "added"):
        value = item.get(key)
        if isinstance(value, str) and value:
            try:
                return _iso_to_timestamp(value)
            except ValueError:
                continue
    return 0.0


def _iso_to_timestamp(value: str) -> float:
    from datetime import datetime

    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).timestamp()
