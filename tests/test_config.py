from __future__ import annotations

from pathlib import Path

from home_llm.config import load_config


def test_load_config_uses_env_dsn(tmp_path: Path, monkeypatch) -> None:
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
api_key_env_var = "HOME_LLM_QDRANT_API_KEY"

[ingest]
source_dirs = ["./docs/*/2026"]
chunk_size = 1400
chunk_overlap = 200
extensions = [".pdf"]

[query]
top_k = 6

[paperless]
enabled = true
base_url = "https://paperless.example.com"
token = "token-value"
token_env_var = "HOME_LLM_PAPERLESS_TOKEN"
page_size = 50
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "HOME_LLM_POSTGRES_DSN", "postgresql://user:pass@db.example.com:5432/home_llm"
    )
    monkeypatch.setenv("HOME_LLM_QDRANT_API_KEY", "qdrant-secret")
    monkeypatch.setenv("HOME_LLM_PAPERLESS_TOKEN", "paperless-secret")

    config = load_config(config_path)

    assert config.postgres.dsn == "postgresql://user:pass@db.example.com:5432/home_llm"
    assert config.qdrant.api_key == "qdrant-secret"
    assert config.paperless.enabled is True
    assert config.paperless.base_url == "https://paperless.example.com"
    assert config.paperless.token == "paperless-secret"
    assert config.paperless.page_size == 50


def test_load_config_resolves_wildcard_source_dirs(tmp_path: Path, monkeypatch) -> None:
    docs_root = tmp_path / "docs" / "bank" / "2026"
    docs_root.mkdir(parents=True)
    monkeypatch.setenv(
        "HOME_LLM_POSTGRES_DSN", "postgresql://user:pass@db.example.com:5432/home_llm"
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
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
source_dirs = ["{tmp_path}/docs/*/2026"]
chunk_size = 1400
chunk_overlap = 200
extensions = [".pdf"]

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

    config = load_config(config_path)

    assert config.ingest.source_dirs == [docs_root.resolve()]


def test_load_config_builds_dsn_from_split_postgres_env_vars(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[postgres]
dsn = ""
dsn_env_var = ""
user = ""
user_env_var = "HOME_LLM_POSTGRES_USER"
host = ""
host_env_var = "HOME_LLM_POSTGRES_HOST"
port = ""
port_env_var = "HOME_LLM_POSTGRES_PORT"
password = ""
password_env_var = "HOME_LLM_POSTGRES_PASSWORD"
database = ""
database_env_var = "HOME_LLM_POSTGRES_DATABASE"

[ollama]
base_url = "http://127.0.0.1:11434"
chat_model = "llama3.1:8b"
embedding_model = "embeddinggemma:latest"
enabled = true

[ingest]
source_dirs = []

[query]
top_k = 6
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME_LLM_POSTGRES_USER", "home_llm_user")
    monkeypatch.setenv("HOME_LLM_POSTGRES_HOST", "db.example.com")
    monkeypatch.setenv("HOME_LLM_POSTGRES_PORT", "5432")
    monkeypatch.setenv("HOME_LLM_POSTGRES_PASSWORD", "p@ss word")
    monkeypatch.setenv("HOME_LLM_POSTGRES_DATABASE", "home_llm")

    config = load_config(config_path)

    assert (
        config.postgres.dsn
        == "postgresql://home_llm_user:p%40ss%20word@db.example.com:5432/home_llm"
    )


def test_load_config_defaults_and_duplicate_source_dirs(tmp_path: Path, monkeypatch) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    monkeypatch.delenv("HOME_LLM_POSTGRES_DSN", raising=False)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[postgres]
dsn = "postgresql://user:pass@db.example.com:5432/home_llm"
dsn_env_var = ""

[ollama]
base_url = "http://127.0.0.1:11434/"
chat_model = "llama3.1:8b"

[ingest]
source_dirs = ["{docs_root}", "{docs_root}"]

[query]
top_k = 7
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.postgres.dsn == "postgresql://user:pass@db.example.com:5432/home_llm"
    assert config.ollama.base_url == "http://127.0.0.1:11434"
    assert config.ollama.embedding_model == "embeddinggemma:latest"
    assert config.qdrant.enabled is False
    assert config.ingest.source_dirs == [docs_root.resolve()]
    assert config.ingest.extensions == {".pdf", ".txt", ".md", ".csv"}
    assert config.paperless.page_size == 100
