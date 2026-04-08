from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from home_llm import llm, paperless_client, query_service, vector_store
from home_llm.config import OllamaConfig, PaperlessConfig, QdrantConfig
from home_llm.models import SearchResult


def make_ollama_config() -> OllamaConfig:
    return OllamaConfig(
        base_url="http://ollama.local",
        chat_model="chat-model",
        embedding_model="embed-model",
        enabled=True,
    )


def make_qdrant_config(api_key: str = "") -> QdrantConfig:
    return QdrantConfig(
        enabled=True,
        base_url="http://qdrant.local",
        collection="docs",
        api_key=api_key,
    )


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: object | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self.raise_called = False

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self) -> None:
        self.raise_called = True
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def test_llm_helpers_and_embed_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    responses = iter(
        [
            FakeResponse(payload={"message": {"content": "  answer text  "}}),
            FakeResponse(payload={"embeddings": [[0.1, 0.2]]}),
            FakeResponse(status_code=404, payload={"error": "model not found"}),
            FakeResponse(payload={"embedding": [0.3, 0.4]}),
        ]
    )

    def fake_post(url: str, json: dict[str, object], timeout: int) -> FakeResponse:
        calls.append((url, json))
        return next(responses)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    results = [
        SearchResult(
            chunk_id=1,
            file_path="/tmp/doc.txt",
            doc_type="statement",
            page_label="page 1",
            content="excerpt",
            score=0.8,
        )
    ]

    assert llm.ask_ollama("question?", results, make_ollama_config()) == "answer text"
    assert llm.embed_texts(["one", "two"], make_ollama_config()) == [[0.1, 0.2], [0.3, 0.4]]
    assert calls[0][0].endswith("/api/chat")
    assert calls[1][0].endswith("/api/embed")
    assert calls[3][0].endswith("/api/embeddings")
    assert llm._build_context([]) == "No relevant excerpts found."
    assert "File: /tmp/doc.txt" in llm._build_context(results)


def test_llm_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *_args, **_kwargs: FakeResponse(payload={"embeddings": []}),
    )
    with pytest.raises(RuntimeError, match="no embeddings"):
        llm.embed_texts(["one"], make_ollama_config())

    responses = iter(
        [
            FakeResponse(status_code=404, payload={"error": "missing"}),
            FakeResponse(status_code=404, payload={"error": "not found"}),
        ]
    )
    monkeypatch.setattr(llm.requests, "post", lambda *_args, **_kwargs: next(responses))
    with pytest.raises(RuntimeError, match='embedding model "embed-model" was not found'):
        llm.embed_texts(["one"], make_ollama_config())

    with pytest.raises(RuntimeError, match='embedding model "embed-model" was not found'):
        llm._raise_ollama_model_error(
            FakeResponse(status_code=404, payload={"error": "Not Found"}),
            "embed-model",
        )

    with pytest.raises(RuntimeError, match="http 500"):
        llm._raise_ollama_model_error(
            FakeResponse(status_code=500, payload=ValueError("bad json"), text="oops"),
            "embed-model",
        )


def test_qdrant_store_behaviors(monkeypatch: pytest.MonkeyPatch) -> None:
    get_calls: list[tuple[str, dict[str, str], int]] = []
    put_calls: list[tuple[str, dict[str, str], object, int]] = []
    post_calls: list[tuple[str, dict[str, str], object, int]] = []

    config = make_qdrant_config(api_key="secret")
    store = vector_store.QdrantStore(config)

    monkeypatch.setattr(
        vector_store.requests,
        "get",
        lambda url, headers, timeout: (
            get_calls.append((url, headers, timeout)) or FakeResponse(status_code=404)
        ),
    )
    monkeypatch.setattr(
        vector_store.requests,
        "put",
        lambda url, headers, json, timeout: (
            put_calls.append((url, headers, json, timeout)) or FakeResponse()
        ),
    )
    monkeypatch.setattr(
        vector_store.requests,
        "post",
        lambda url, headers, json, timeout: (
            post_calls.append((url, headers, json, timeout))
            or FakeResponse(
                payload={
                    "result": [
                        {
                            "payload": {
                                "file_path": "/tmp/a",
                                "doc_type": "bank",
                                "page_label": "page 1",
                                "content": "body",
                            },
                            "score": 0.9,
                        }
                    ]
                }
            )
        ),
    )

    store.ensure_collection(3)
    store.replace_document_chunks("/tmp/a", [])
    store.replace_document_chunks("/tmp/a", [{"id": "1", "vector": [1.0], "payload": {}}])
    store.delete_document("/tmp/a")
    results = store.search([0.1, 0.2], 5)

    assert store.headers["api-key"] == "secret"
    assert get_calls and put_calls and post_calls
    assert results[0].file_path == "/tmp/a"
    assert vector_store.build_point_id(
        "/tmp/a",
        "page 1",
        0,
    ) == vector_store.build_point_id("/tmp/a", "page 1", 0)

    monkeypatch.setattr(
        vector_store.requests,
        "post",
        lambda *_args, **_kwargs: FakeResponse(status_code=500),
    )
    with pytest.raises(RuntimeError, match="http 500"):
        store.delete_document("/tmp/a")

    monkeypatch.setattr(
        vector_store.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(status_code=200),
    )
    monkeypatch.setattr(
        vector_store.requests,
        "put",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not create collection")
        ),
    )
    store.ensure_collection(5)
    assert "api-key" not in vector_store.QdrantStore(make_qdrant_config(api_key="")).headers


def test_paperless_client_and_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="base_url"):
        paperless_client.PaperlessClient(PaperlessConfig(True, "", "token", 10))
    with pytest.raises(ValueError, match="token"):
        paperless_client.PaperlessClient(PaperlessConfig(True, "http://paperless", "", 10))

    payloads = iter(
        [
            {
                "results": [
                    {
                        "id": 1,
                        "title": "Mortgage Statement",
                        "archive_serial_number": "",
                        "storage_path": "mortgage/2026",
                        "content": "balance is here",
                        "modified": "2026-01-02T00:00:00Z",
                        "tags": [1],
                    },
                    {"id": 2, "content": "   "},
                ],
                "next": "http://paperless/api/documents/?page=2",
            },
            {
                "results": [
                    {
                        "id": 3,
                        "title": "Other",
                        "archive_serial_number": "ARCHIVE",
                        "storage_path": "misc",
                        "content": "usable",
                        "added": "2026-01-03T00:00:00Z",
                    }
                ],
                "next": None,
            },
        ]
    )

    class FakeSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def get(self, url: str, timeout: int) -> FakeResponse:
            return FakeResponse(payload=next(payloads))

    monkeypatch.setattr(paperless_client.requests, "Session", FakeSession)
    client = paperless_client.PaperlessClient(
        PaperlessConfig(enabled=True, base_url="http://paperless", token="abc", page_size=25)
    )
    documents = client.iter_documents()

    assert client.session.headers["Authorization"] == "Token abc"
    assert len(documents) == 2
    assert documents[0].chunks[0].doc_type == "mortgage"
    assert documents[0].chunks[0].file_path.endswith(
        "/mortgage/2026/1/Mortgage Statement.paperless.txt"
    )
    assert documents[1].chunks[0].file_path.endswith("/misc/3/ARCHIVE")
    assert paperless_client._detect_doc_type({"title": "retirement overview"}) == "retirement"
    assert paperless_client._detect_doc_type({"title": "misc"}) == "paperless"
    assert (
        paperless_client._parse_modified_at({"modified": "bad", "added": "2026-01-05T00:00:00Z"})
        > 0
    )
    assert paperless_client._parse_modified_at({}) == 0.0
    assert paperless_client._iso_to_timestamp("2026-01-01T00:00:00Z") > 0


@dataclass
class DummyStore:
    closed: bool = False
    search_result: list[SearchResult] | None = None
    fact_rows: list[dict[str, object]] | None = None
    stats_result: dict[str, int] | None = None

    def close(self) -> None:
        self.closed = True

    def search(self, query: str, limit: int) -> list[SearchResult]:
        assert query and limit
        return self.search_result or []

    def list_facts(self, limit: int) -> list[dict[str, object]]:
        assert limit
        return self.fact_rows or []

    def stats(self) -> dict[str, int]:
        return self.stats_result or {"documents": 0, "chunks": 0, "facts": 0}


def make_app_config(qdrant_enabled: bool = False, ollama_enabled: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        postgres=SimpleNamespace(dsn="postgres://dsn"),
        ollama=SimpleNamespace(enabled=ollama_enabled),
        qdrant=SimpleNamespace(enabled=qdrant_enabled),
        query=SimpleNamespace(top_k=6),
    )


def test_query_service_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    result = SearchResult(1, "/tmp/file.txt", "bank", "page 1", "content", 0.7)

    store_one = DummyStore(search_result=[])
    monkeypatch.setattr(query_service, "Store", lambda _dsn: store_one)
    payload = query_service.ask_question(make_app_config(), "question")
    assert payload["answer"] is None
    assert payload["message"].startswith("No matching excerpts")
    assert store_one.closed is True

    store_two = DummyStore(search_result=[result])
    monkeypatch.setattr(query_service, "Store", lambda _dsn: store_two)
    monkeypatch.setattr(query_service, "ask_ollama", lambda *_args: "answer")
    payload = query_service.ask_question(make_app_config(ollama_enabled=True), "question", top_k=2)
    assert payload["answer"] == "answer"
    assert payload["results"][0]["file_path"] == "/tmp/file.txt"

    store_three = DummyStore(search_result=[result])
    monkeypatch.setattr(query_service, "Store", lambda _dsn: store_three)
    rows = query_service.search_excerpts(make_app_config(), "needle")
    assert rows[0]["content"] == "content"

    store_three_b = DummyStore(search_result=[result])
    monkeypatch.setattr(query_service, "Store", lambda _dsn: store_three_b)
    payload = query_service.ask_question(make_app_config(ollama_enabled=False), "question")
    assert payload["answer"] is None
    assert payload["message"] is None

    store_four = DummyStore(
        fact_rows=[
            {
                "file_path": "/tmp/path/report.pdf",
                "doc_type": "statement",
                "fact_type": "mentioned_amount",
                "fact_value": "$4.00",
                "page_label": "full",
            }
        ]
    )
    monkeypatch.setattr(query_service, "Store", lambda _dsn: store_four)
    facts = query_service.list_facts(make_app_config(), limit=10)
    assert facts[0]["file_name"] == "report.pdf"

    store_five = DummyStore(stats_result={"documents": 1, "chunks": 2, "facts": 3})
    monkeypatch.setattr(query_service, "Store", lambda _dsn: store_five)
    assert query_service.get_stats(make_app_config()) == {"documents": 1, "chunks": 2, "facts": 3}

    monkeypatch.setattr(query_service, "embed_texts", lambda *_args: [[0.1, 0.2]])

    class FakeQdrant:
        def __init__(self, _config: object) -> None:
            pass

        def search(self, embedding: list[float], limit: int) -> list[SearchResult]:
            assert embedding == [0.1, 0.2]
            assert limit == 4
            return [result]

    monkeypatch.setattr(query_service, "QdrantStore", FakeQdrant)
    assert query_service._retrieve_results(
        DummyStore(),
        make_app_config(qdrant_enabled=True),
        "q",
        4,
    ) == [result]
