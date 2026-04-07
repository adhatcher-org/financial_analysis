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
page_size = 50
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "HOME_LLM_POSTGRES_DSN", "postgresql://user:pass@db.example.com:5432/home_llm"
    )

    config = load_config(config_path)

    assert config.postgres.dsn == "postgresql://user:pass@db.example.com:5432/home_llm"
    assert config.paperless.enabled is True
    assert config.paperless.base_url == "https://paperless.example.com"
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
