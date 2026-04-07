from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from home_llm import ingest, store
from home_llm.models import DocumentChunk, SearchResult


class FakeStore:
    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self.upserts: list[tuple[str, str, float]] = []
        self.cleared: list[int] = []
        self.inserted_chunks: list[tuple[int, str, str]] = []
        self.inserted_facts: list[tuple[int, str, str, str]] = []

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True

    def stats(self) -> dict[str, int]:
        return {"documents": 9, "chunks": 8, "facts": 7}

    def upsert_document(self, file_path: str, doc_type: str, modified_at: float) -> int:
        self.upserts.append((file_path, doc_type, modified_at))
        return 42

    def clear_document_contents(self, document_id: int) -> None:
        self.cleared.append(document_id)

    def insert_chunk(self, document_id: int, page_label: str, content: str) -> int:
        self.inserted_chunks.append((document_id, page_label, content))
        return len(self.inserted_chunks)

    def insert_fact(
        self,
        document_id: int,
        fact_type: str,
        fact_value: str,
        page_label: str,
    ) -> None:
        self.inserted_facts.append((document_id, fact_type, fact_value, page_label))


class FakeQdrant:
    def __init__(self) -> None:
        self.collection_sizes: list[int] = []
        self.replacements: list[tuple[str, list[dict[str, object]]]] = []

    def ensure_collection(self, vector_size: int) -> None:
        self.collection_sizes.append(vector_size)

    def replace_document_chunks(self, file_path: str, points: list[dict[str, object]]) -> None:
        self.replacements.append((file_path, points))


def make_config(
    *,
    qdrant_enabled: bool = False,
    paperless_enabled: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        postgres=SimpleNamespace(dsn="postgres://dsn"),
        qdrant=SimpleNamespace(enabled=qdrant_enabled),
        ollama=SimpleNamespace(enabled=True),
        paperless=SimpleNamespace(enabled=paperless_enabled),
        ingest=SimpleNamespace(chunk_size=5, chunk_overlap=1, source_dirs=[], extensions={".txt"}),
        query=SimpleNamespace(top_k=3),
    )


def test_store_document_and_iter_source_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_store = FakeStore()
    fake_qdrant = FakeQdrant()
    pages = [DocumentChunk("/tmp/a.txt", "bank", "page 1", "abcdef")]

    monkeypatch.setattr(ingest, "chunk_text", lambda *_args: ["chunk-one", "chunk-two"])
    monkeypatch.setattr(ingest, "embed_texts", lambda *_args: [[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr(
        ingest,
        "extract_facts",
        lambda page: [
            {
                "fact_type": "kind",
                "fact_value": page.doc_type,
                "page_label": page.page_label,
            }
        ],
    )
    monkeypatch.setattr(
        ingest,
        "build_point_id",
        lambda file_path, page_label, index: f"{file_path}:{page_label}:{index}",
    )

    ingest._store_document(
        store=fake_store,
        qdrant=fake_qdrant,
        file_path="/tmp/a.txt",
        doc_type="bank",
        modified_at=1.5,
        pages=pages,
        config=make_config(qdrant_enabled=True),
    )

    assert fake_store.upserts == [("/tmp/a.txt", "bank", 1.5)]
    assert fake_store.cleared == [42]
    assert len(fake_store.inserted_chunks) == 2
    assert fake_store.inserted_facts == [(42, "kind", "bank", "page 1")]
    assert fake_qdrant.collection_sizes == [2]
    assert fake_qdrant.replacements[0][0] == "/tmp/a.txt"
    assert len(fake_qdrant.replacements[0][1]) == 2

    existing = tmp_path / "docs"
    existing.mkdir()
    file_one = existing / "a.txt"
    file_one.write_text("a", encoding="utf-8")
    nested = existing / "nested"
    nested.mkdir()
    file_two = nested / "b.txt"
    file_two.write_text("b", encoding="utf-8")
    ignored = existing / "ignored.md"
    ignored.write_text("x", encoding="utf-8")

    files = list(ingest.iter_source_files([tmp_path / "missing", existing, existing], {".txt"}))
    assert files == [file_one.resolve(), file_two.resolve()]

    no_qdrant_store = FakeStore()
    monkeypatch.setattr(ingest, "chunk_text", lambda *_args: ["chunk"])
    monkeypatch.setattr(ingest, "extract_facts", lambda _page: [])
    ingest._store_document(
        store=no_qdrant_store,
        qdrant=None,
        file_path="/tmp/empty.txt",
        doc_type="bank",
        modified_at=2.0,
        pages=[DocumentChunk("/tmp/empty.txt", "bank", "page 1", "ignored")],
        config=make_config(),
    )
    assert no_qdrant_store.inserted_chunks == [(42, "page 1", "chunk")]


def test_ingest_local_and_paperless_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_store = FakeStore()
    fake_qdrant = FakeQdrant()
    file_ok = tmp_path / "ok.txt"
    file_ok.write_text("ok", encoding="utf-8")
    file_empty = tmp_path / "empty.txt"
    file_empty.write_text("empty", encoding="utf-8")
    file_bad = tmp_path / "bad.txt"
    file_bad.write_text("bad", encoding="utf-8")

    monkeypatch.setattr(ingest, "iter_source_files", lambda *_args: [file_bad, file_empty, file_ok])

    def fake_read_document(path: Path) -> list[DocumentChunk]:
        if path == file_bad:
            raise RuntimeError("broken")
        if path == file_empty:
            return []
        return [DocumentChunk(str(path), "bank", "full", "content")]

    stored: list[str] = []
    monkeypatch.setattr(ingest, "read_document", fake_read_document)
    monkeypatch.setattr(
        ingest,
        "_store_document",
        lambda **kwargs: stored.append(kwargs["file_path"]),
    )

    processed, skipped, errors = ingest._ingest_local_documents(
        fake_store,
        fake_qdrant,
        make_config(),
    )
    assert (processed, skipped) == (1, 2)
    assert stored == [str(file_ok)]
    assert any("broken" in item for item in errors)
    assert any("no extractable text found" in item for item in errors)

    class FakePaperlessClient:
        def __init__(self, _config: object) -> None:
            pass

        def iter_documents(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(document_id=1, modified_at=1.0, chunks=[]),
                SimpleNamespace(
                    document_id=2,
                    modified_at=2.0,
                    chunks=[
                        DocumentChunk(
                            "/paperless/doc.txt",
                            "paperless",
                            "paperless-content",
                            "body",
                        )
                    ],
                ),
            ]

    monkeypatch.setattr(ingest, "PaperlessClient", FakePaperlessClient)
    stored.clear()
    processed, skipped, errors = ingest._ingest_paperless_documents(
        fake_store,
        fake_qdrant,
        make_config(),
    )
    assert (processed, skipped) == (1, 1)
    assert stored == ["/paperless/doc.txt"]
    assert errors == ["paperless:1: no extractable text found"]


def test_ingest_documents_commit_and_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    local_store = FakeStore()
    monkeypatch.setattr(ingest, "Store", lambda _dsn: local_store)
    monkeypatch.setattr(ingest, "_ingest_local_documents", lambda *_args: (2, 1, ["warn"]))
    monkeypatch.setattr(ingest, "QdrantStore", lambda _config: FakeQdrant())

    result = ingest.ingest_documents(make_config(qdrant_enabled=True))

    assert result["processed"] == 2
    assert result["skipped"] == 1
    assert result["errors"] == ["warn"]
    assert local_store.committed is True
    assert local_store.closed is True

    failing_store = FakeStore()
    monkeypatch.setattr(ingest, "Store", lambda _dsn: failing_store)
    monkeypatch.setattr(
        ingest,
        "_ingest_local_documents",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        ingest.ingest_documents(make_config())

    assert failing_store.rolled_back is True
    assert failing_store.closed is True


class FakeCursor:
    def __init__(self, row: object = None, rows: list[dict[str, object]] | None = None) -> None:
        self._row = row
        self._rows = rows or []

    def fetchone(self) -> object:
        return self._row

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False
        self.committed = False
        self.rolled_back = False
        self.calls: list[tuple[str, object]] = []
        self.rows_by_sql: list[FakeCursor] = []

    def execute(self, sql: str, params: object = None) -> FakeCursor:
        self.calls.append((sql, params))
        if self.rows_by_sql:
            return self.rows_by_sql.pop(0)
        return FakeCursor()

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def test_store_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = FakeConnection()
    monkeypatch.setattr(store, "connect", lambda dsn, row_factory: connection)
    instance = store.Store("postgres://dsn")

    assert connection.committed is True
    assert store.SCHEMA.strip().startswith("CREATE TABLE")

    instance.connection.rows_by_sql = [FakeCursor({"id": 5})]
    assert instance.upsert_document("/tmp/a", "bank", 1.0) == 5

    instance.connection.rows_by_sql = [FakeCursor(None)]
    with pytest.raises(RuntimeError, match="Failed to upsert"):
        instance.upsert_document("/tmp/a", "bank", 1.0)

    instance.connection.rows_by_sql = [FakeCursor({"id": 7})]
    assert instance.insert_chunk(1, "page 1", "body") == 7

    instance.connection.rows_by_sql = [FakeCursor(None)]
    with pytest.raises(RuntimeError, match="Failed to insert chunk"):
        instance.insert_chunk(1, "page 1", "body")

    instance.clear_document_contents(1)
    instance.insert_fact(1, "kind", "value", "page 1")
    instance.commit()
    instance.rollback()
    assert connection.committed is True
    assert connection.rolled_back is True

    instance.connection.rows_by_sql = [
        FakeCursor(
            rows=[
                {
                    "chunk_id": 1,
                    "file_path": "/tmp/a",
                    "doc_type": "bank",
                    "page_label": "page 1",
                    "content": "body",
                    "score": 0.5,
                }
            ]
        )
    ]
    search_rows = instance.search("query", 4)
    assert search_rows == [SearchResult(1, "/tmp/a", "bank", "page 1", "body", 0.5)]

    instance.connection.rows_by_sql = [FakeCursor(rows=[{"file_path": "/tmp/a"}])]
    assert instance.list_facts() == [{"file_path": "/tmp/a"}]

    instance.connection.rows_by_sql = [FakeCursor(rows=[{"file_path": "/tmp/b"}])]
    assert instance.list_facts(limit=3) == [{"file_path": "/tmp/b"}]

    instance.connection.rows_by_sql = [
        FakeCursor({"count": 1}),
        FakeCursor({"count": 2}),
        FakeCursor({"count": 3}),
    ]
    assert instance.stats() == {"documents": 1, "chunks": 2, "facts": 3}

    instance.connection.rows_by_sql = [
        FakeCursor(None),
        FakeCursor({"count": 2}),
        FakeCursor({"count": 3}),
    ]
    with pytest.raises(RuntimeError, match="Failed to fetch store stats"):
        instance.stats()

    assert store._row_to_result(
        {
            "chunk_id": "9",
            "file_path": Path("/tmp/x"),
            "doc_type": "statement",
            "page_label": "full",
            "content": "snippet",
            "score": "0.75",
        }
    ) == SearchResult(9, "/tmp/x", "statement", "full", "snippet", 0.75)

    instance.close()
    assert connection.closed is True
