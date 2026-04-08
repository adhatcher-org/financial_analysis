from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

from home_llm.config import AppConfig, load_config
from home_llm.query_service import ask_question, get_stats, list_facts, search_excerpts


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=25)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Home LLM API",
        version="0.1.0",
        description=(
            "FastAPI service for querying financial and insurance documents indexed by Home LLM."
        ),
    )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "Home LLM API",
            "status": "ok",
            "docs": "/docs",
            "endpoints": {
                "health": "/health",
                "ask": "/ask",
                "facts": "/facts",
                "search_get": "/search",
                "search_post": "/search",
            },
        }

    @app.get("/favicon.ico", status_code=204)
    @app.get("/apple-touch-icon.png", status_code=204)
    @app.get("/apple-touch-icon-precomposed.png", status_code=204)
    def browser_icon() -> Response:
        return Response(status_code=204)

    @app.get("/health")
    def health() -> dict[str, Any]:
        config = require_config()
        stats = get_stats(config)
        return {
            "status": "ok",
            "qdrant_enabled": config.qdrant.enabled,
            "ollama_enabled": config.ollama.enabled,
            "stats": stats,
        }

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        config = require_config()
        try:
            return ask_question(config, request.question, request.top_k)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/facts")
    def facts(limit: int = Query(default=50, ge=1, le=1000)) -> dict[str, Any]:
        config = require_config()
        try:
            rows = list_facts(config, limit=limit)
            return {"items": rows, "count": len(rows)}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/search")
    def search(
        query: str = Query(..., min_length=1),
        top_k: int = Query(default=10, ge=1, le=50),
    ) -> dict[str, Any]:
        config = require_config()
        try:
            rows = search_excerpts(config, query=query, top_k=top_k)
            return {"items": rows, "count": len(rows)}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/search")
    def search_post(request: SearchRequest) -> dict[str, Any]:
        config = require_config()
        try:
            rows = search_excerpts(config, query=request.query, top_k=request.top_k)
            return {"items": rows, "count": len(rows)}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    config_path = os.environ.get("HOME_LLM_CONFIG", "config.toml")
    return load_config(config_path)


def require_config() -> AppConfig:
    try:
        return get_config()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app = create_app()
