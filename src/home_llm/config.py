from __future__ import annotations

import glob
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


@dataclass(slots=True)
class PostgresConfig:
    dsn: str


@dataclass(slots=True)
class OllamaConfig:
    base_url: str
    chat_model: str
    embedding_model: str
    enabled: bool


@dataclass(slots=True)
class QdrantConfig:
    enabled: bool
    base_url: str
    collection: str
    api_key: str


@dataclass(slots=True)
class IngestConfig:
    source_dirs: list[Path]
    chunk_size: int
    chunk_overlap: int
    extensions: set[str]


@dataclass(slots=True)
class PaperlessConfig:
    enabled: bool
    base_url: str
    token: str
    page_size: int


@dataclass(slots=True)
class QueryConfig:
    top_k: int


@dataclass(slots=True)
class AppConfig:
    postgres: PostgresConfig
    ollama: OllamaConfig
    qdrant: QdrantConfig
    ingest: IngestConfig
    paperless: PaperlessConfig
    query: QueryConfig


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    postgres = raw.get("postgres", {})
    ollama = raw["ollama"]
    qdrant = raw.get("qdrant", {})
    ingest = raw["ingest"]
    paperless = raw.get("paperless", {})
    query = raw["query"]

    return AppConfig(
        postgres=PostgresConfig(
            dsn=_resolve_postgres_dsn(postgres),
        ),
        ollama=OllamaConfig(
            base_url=ollama["base_url"].rstrip("/"),
            chat_model=ollama["chat_model"],
            embedding_model=ollama.get("embedding_model", "embeddinggemma:latest"),
            enabled=bool(ollama.get("enabled", True)),
        ),
        qdrant=QdrantConfig(
            enabled=bool(qdrant.get("enabled", False)),
            base_url=str(qdrant.get("base_url", "http://127.0.0.1:6333")).rstrip("/"),
            collection=str(qdrant.get("collection", "financial_documents")),
            api_key=_resolve_secret(
                qdrant,
                value_key="api_key",
                env_var_key="api_key_env_var",
                default_env_var="HOME_LLM_QDRANT_API_KEY",
            ),
        ),
        ingest=IngestConfig(
            source_dirs=_resolve_source_dirs(ingest["source_dirs"]),
            chunk_size=int(ingest.get("chunk_size", 1400)),
            chunk_overlap=int(ingest.get("chunk_overlap", 200)),
            extensions={
                str(ext).lower()
                for ext in ingest.get("extensions", [".pdf", ".txt", ".md", ".csv"])
            },
        ),
        paperless=PaperlessConfig(
            enabled=bool(paperless.get("enabled", False)),
            base_url=str(paperless.get("base_url", "")).rstrip("/"),
            token=_resolve_secret(
                paperless,
                value_key="token",
                env_var_key="token_env_var",
                default_env_var="HOME_LLM_PAPERLESS_TOKEN",
            ),
            page_size=int(paperless.get("page_size", 100)),
        ),
        query=QueryConfig(top_k=int(query.get("top_k", 6))),
    )


def _resolve_source_dirs(items: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for item in items:
        expanded = str(Path(item).expanduser())
        matches = glob.glob(expanded)
        paths = matches if matches else [expanded]
        for candidate in paths:
            path = Path(candidate).resolve()
            if path in seen:
                continue
            seen.add(path)
            resolved.append(path)
    return resolved


def _resolve_postgres_dsn(raw: dict[str, str]) -> str:
    dsn = _resolve_secret(
        raw,
        value_key="dsn",
        env_var_key="dsn_env_var",
        default_env_var="HOME_LLM_POSTGRES_DSN",
    )
    if not dsn:
        dsn = _build_postgres_dsn(raw)
    if not dsn:
        raise ValueError(
            "PostgreSQL DSN is not configured. "
            "Set [postgres].dsn in config.toml, define HOME_LLM_POSTGRES_DSN, "
            "or provide HOME_LLM_POSTGRES_USER/HOST/PORT/PASSWORD/DATABASE."
        )
    return dsn


def _resolve_secret(
    raw: dict[str, str],
    *,
    value_key: str,
    env_var_key: str,
    default_env_var: str,
) -> str:
    value = str(raw.get(value_key, "")).strip()
    env_var = str(raw.get(env_var_key, default_env_var)).strip()
    if env_var:
        value = os.environ.get(env_var, value).strip()
    return value


def _build_postgres_dsn(raw: dict[str, str]) -> str:
    user = _resolve_secret(
        raw,
        value_key="user",
        env_var_key="user_env_var",
        default_env_var="HOME_LLM_POSTGRES_USER",
    )
    host = _resolve_secret(
        raw,
        value_key="host",
        env_var_key="host_env_var",
        default_env_var="HOME_LLM_POSTGRES_HOST",
    )
    port = _resolve_secret(
        raw,
        value_key="port",
        env_var_key="port_env_var",
        default_env_var="HOME_LLM_POSTGRES_PORT",
    )
    password = _resolve_secret(
        raw,
        value_key="password",
        env_var_key="password_env_var",
        default_env_var="HOME_LLM_POSTGRES_PASSWORD",
    )
    database = _resolve_secret(
        raw,
        value_key="database",
        env_var_key="database_env_var",
        default_env_var="HOME_LLM_POSTGRES_DATABASE",
    )

    if not all([user, host, port, password, database]):
        return ""

    quoted_user = quote(user, safe="")
    quoted_password = quote(password, safe="")
    quoted_database = quote(database, safe="")
    return (
        f"postgresql://{quoted_user}:{quoted_password}@{host}:{port}/{quoted_database}"
    )
