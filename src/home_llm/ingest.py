from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from home_llm.config import AppConfig
from home_llm.document_parsers import read_document
from home_llm.fact_extractor import extract_facts
from home_llm.llm import embed_texts
from home_llm.models import DocumentChunk
from home_llm.paperless_client import PaperlessClient
from home_llm.store import Store
from home_llm.text_utils import chunk_text
from home_llm.vector_store import QdrantStore, build_point_id


def ingest_documents(config: AppConfig) -> dict[str, Any]:
    store = Store(config.postgres.dsn)
    processed = 0
    skipped = 0
    errors: list[str] = []

    try:
        qdrant = QdrantStore(config.qdrant) if config.qdrant.enabled else None
        ingest_result = (
            _ingest_paperless_documents(store, qdrant, config)
            if config.paperless.enabled
            else _ingest_local_documents(store, qdrant, config)
        )
        processed, skipped, errors = ingest_result

        store.commit()
        stats: dict[str, Any] = store.stats()
        stats["processed"] = processed
        stats["skipped"] = skipped
        stats["errors"] = errors
        return stats
    except Exception:
        store.rollback()
        raise
    finally:
        store.close()


def _ingest_local_documents(
    store: Store,
    qdrant: QdrantStore | None,
    config: AppConfig,
) -> tuple[int, int, list[str]]:
    processed = 0
    skipped = 0
    errors: list[str] = []
    for file_path in iter_source_files(config.ingest.source_dirs, config.ingest.extensions):
        try:
            parsed_pages = read_document(file_path)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            errors.append(f"{file_path}: {exc}")
            continue
        if not parsed_pages:
            skipped += 1
            errors.append(f"{file_path}: no extractable text found")
            continue

        _store_document(
            store=store,
            qdrant=qdrant,
            file_path=str(file_path),
            doc_type=parsed_pages[0].doc_type,
            modified_at=file_path.stat().st_mtime,
            pages=parsed_pages,
            config=config,
        )
        processed += 1
    return processed, skipped, errors


def _ingest_paperless_documents(
    store: Store,
    qdrant: QdrantStore | None,
    config: AppConfig,
) -> tuple[int, int, list[str]]:
    processed = 0
    skipped = 0
    errors: list[str] = []
    client = PaperlessClient(config.paperless)
    documents = client.iter_documents()
    for document in documents:
        if not document.chunks:
            skipped += 1
            errors.append(f"paperless:{document.document_id}: no extractable text found")
            continue
        _store_document(
            store=store,
            qdrant=qdrant,
            file_path=document.chunks[0].file_path,
            doc_type=document.chunks[0].doc_type,
            modified_at=document.modified_at,
            pages=document.chunks,
            config=config,
        )
        processed += 1
    return processed, skipped, errors


def _store_document(
    *,
    store: Store,
    qdrant: QdrantStore | None,
    file_path: str,
    doc_type: str,
    modified_at: float,
    pages: list[DocumentChunk],
    config: AppConfig,
) -> None:
    document_id = store.upsert_document(
        file_path=file_path,
        doc_type=doc_type,
        modified_at=modified_at,
    )
    store.clear_document_contents(document_id)
    qdrant_points = []

    for page in pages:
        chunks = chunk_text(page.content, config.ingest.chunk_size, config.ingest.chunk_overlap)
        embeddings = embed_texts(chunks, config.ollama) if qdrant and chunks else []
        if qdrant and embeddings:
            qdrant.ensure_collection(vector_size=len(embeddings[0]))
        for index, chunk in enumerate(chunks):
            store.insert_chunk(document_id, page.page_label, chunk)
            if qdrant:
                qdrant_points.append(
                    {
                        "id": build_point_id(file_path, page.page_label, index),
                        "vector": embeddings[index],
                        "payload": {
                            "file_path": file_path,
                            "doc_type": page.doc_type,
                            "page_label": page.page_label,
                            "content": chunk,
                        },
                    }
                )
        for fact in extract_facts(page):
            store.insert_fact(
                document_id,
                fact["fact_type"],
                fact["fact_value"],
                fact["page_label"],
            )

    if qdrant:
        qdrant.replace_document_chunks(file_path, qdrant_points)


def iter_source_files(source_dirs: list[Path], extensions: set[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in extensions:
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield resolved
