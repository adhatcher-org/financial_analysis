from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from home_llm import api
from home_llm.api import create_app
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


def test_api_endpoints_success_and_error_paths(monkeypatch) -> None:
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

    client = TestClient(create_app())

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
        "list_facts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("facts failed")),
    )
    monkeypatch.setattr(
        api,
        "search_excerpts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("search failed")),
    )

    assert client.post("/ask", json={"question": "What?"}).json()["detail"] == "ask failed"
    assert client.get("/facts").json()["detail"] == "facts failed"
    assert client.get("/search", params={"query": "needle"}).json()["detail"] == "search failed"
    assert client.post("/search", json={"query": "needle"}).json()["detail"] == "search failed"
