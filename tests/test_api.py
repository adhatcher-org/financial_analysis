from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from home_llm import api
from home_llm.api import INTERNAL_SERVER_ERROR_DETAIL, create_app
from home_llm.config import (
    AppConfig,
    IngestConfig,
    OllamaConfig,
    PaperlessConfig,
    PostgresConfig,
    QdrantConfig,
    QueryConfig,
)


def test_health_returns_config_error_when_dsn_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[postgres]
dsn = ""
dsn_env_var = "HOME_LLM_POSTGRES_DSN"

[ollama]
base_url = "http://127.0.0.1:11434"
chat_model = "llama3.1:8b"
embedding_model = "embeddinggemma:latest"
enabled = true

[qdrant]
enabled = false
base_url = "http://127.0.0.1:6333"
collection = "financial_documents"
api_key = ""

[ingest]
source_dirs = []
chunk_size = 1400
chunk_overlap = 200
extensions = [".pdf", ".txt", ".md", ".csv"]

[query]
top_k = 6

[paperless]
enabled = false
base_url = ""
token = ""
page_size = 100
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv("HOME_LLM_POSTGRES_DSN", raising=False)
    monkeypatch.setenv("HOME_LLM_CONFIG", str(config_path))
    api.get_config.cache_clear()
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 500
    assert "PostgreSQL DSN is not configured" in response.json()["detail"]


def test_api_endpoints_success_and_error_paths(monkeypatch, caplog) -> None:
    config = AppConfig(
        postgres=PostgresConfig(dsn="postgres://dsn"),
        ollama=OllamaConfig(
            base_url="http://ollama",
            chat_model="chat",
            embedding_model="embed",
            enabled=True,
        ),
        qdrant=QdrantConfig(
            enabled=True,
            base_url="http://qdrant",
            collection="docs",
            api_key="",
        ),
        ingest=IngestConfig(
            source_dirs=[],
            chunk_size=100,
            chunk_overlap=10,
            extensions={".txt"},
        ),
        paperless=PaperlessConfig(
            enabled=False,
            base_url="",
            token="",
            page_size=100,
        ),
        query=QueryConfig(top_k=6),
    )
    monkeypatch.setattr(api, "require_config", lambda: config)
    monkeypatch.setattr(api, "get_stats", lambda _cfg: {"documents": 1, "chunks": 2, "facts": 3})
    monkeypatch.setattr(
        api,
        "ingest_documents",
        lambda _cfg: {
            "processed": 2,
            "skipped": 1,
            "documents": 5,
            "chunks": 8,
            "facts": 13,
            "errors": [],
        },
    )
    monkeypatch.setattr(
        api,
        "ask_question",
        lambda _cfg, question, top_k: {"question": question, "top_k": top_k},
    )
    monkeypatch.setattr(
        api,
        "list_facts",
        lambda _cfg, limit: [
            {
                "file_path": "/tmp/a",
                "fact_type": "kind",
                "fact_value": "value",
                "doc_type": "bank",
                "page_label": "page 1",
                "limit": limit,
            }
        ],
    )
    monkeypatch.setattr(
        api,
        "search_excerpts",
        lambda _cfg, query, top_k: [{"file_path": "/tmp/a", "content": query, "top_k": top_k}],
    )

    class FakeJobManager:
        def start_job(self) -> tuple[dict[str, object], bool]:
            return (
                {
                    "job_id": "job-123",
                    "status": "running",
                    "source": "paperless",
                    "total": 4,
                    "processed": 1,
                    "skipped": 1,
                    "current_index": 3,
                    "current_document": "Mortgage Statement",
                    "current_file_path": "/paperless/1/doc.txt",
                    "started_at": "2026-04-07 09:00:00",
                    "finished_at": None,
                    "error": None,
                    "result": None,
                    "events": [
                        {
                            "kind": "document",
                            "state": "processed",
                            "source": "paperless",
                            "index": 1,
                            "total": 4,
                            "label": "Mortgage Statement",
                            "file_path": "/paperless/1/doc.txt",
                            "timestamp": "2026-04-07 09:00:01",
                        }
                    ],
                },
                True,
            )

        def get_current_job(self) -> dict[str, object]:
            return {"job_id": "job-123", "status": "running", "events": []}

        def get_job(self, job_id: str) -> dict[str, object] | None:
            if job_id == "job-123":
                return {"job_id": "job-123", "status": "completed", "events": []}
            return None

    monkeypatch.setattr(api, "INGEST_JOB_MANAGER", FakeJobManager())

    client = TestClient(create_app())

    root_response = client.get("/")
    documents_response = client.get("/documents")
    favicon_response = client.get("/favicon.ico")
    apple_touch_response = client.get("/apple-touch-icon.png")
    ingest_response = client.post("/ingest")
    ingest_job_response = client.post("/ingest/jobs")

    assert root_response.status_code == 200
    assert "Home LLM" in root_response.text
    assert documents_response.status_code == 200
    assert "Document Management" in documents_response.text
    assert favicon_response.status_code == 204
    assert apple_touch_response.status_code == 204
    assert ingest_response.json()["processed"] == 2
    assert ingest_job_response.json()["started"] is True
    assert ingest_job_response.json()["job"]["job_id"] == "job-123"
    assert client.get("/ingest/jobs/current").json()["job"]["job_id"] == "job-123"
    assert client.get("/ingest/jobs/job-123").json()["job"]["status"] == "completed"
    assert client.get("/ingest/jobs/missing").status_code == 404
    assert client.get("/health").json()["stats"] == {"documents": 1, "chunks": 2, "facts": 3}
    assert client.post("/ask", json={"question": "What?", "top_k": 3}).json() == {
        "question": "What?",
        "top_k": 3,
    }
    assert client.get("/facts", params={"limit": 5}).json()["items"][0]["limit"] == 5
    assert (
        client.get("/search", params={"query": "needle", "top_k": 4}).json()["items"][0]["top_k"]
        == 4
    )
    assert (
        client.post("/search", json={"query": "term", "top_k": 2}).json()["items"][0]["content"]
        == "term"
    )

    monkeypatch.setattr(
        api,
        "ask_question",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ask failed")),
    )
    monkeypatch.setattr(
        api,
        "ingest_documents",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ingest failed")),
    )
    monkeypatch.setattr(
        api,
        "list_facts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("facts failed")),
    )
    monkeypatch.setattr(
        api,
        "search_excerpts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("search failed")),
    )

    responses = [
        client.post("/ask", json={"question": "What?"}),
        client.post("/ingest"),
        client.get("/facts"),
        client.get("/search", params={"query": "needle"}),
        client.post("/search", json={"query": "needle"}),
    ]

    assert all(response.status_code == 500 for response in responses)
    assert all(response.json()["detail"] == INTERNAL_SERVER_ERROR_DETAIL for response in responses)
    assert all(
        message not in response.text
        for response, message in zip(
            responses,
            ["ask failed", "ingest failed", "facts failed", "search failed", "search failed"],
            strict=True,
        )
    )
    assert sum(record.exc_info is not None for record in caplog.records) == 5
