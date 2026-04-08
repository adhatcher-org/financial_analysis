from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from home_llm.config import AppConfig, load_config
from home_llm.ingest import ingest_documents
from home_llm.query_service import ask_question, get_stats, list_facts, search_excerpts

APP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Home LLM</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255, 252, 247, 0.78);
      --panel-strong: #fffaf2;
      --ink: #1f1a16;
      --muted: #6b6258;
      --accent: #b85c38;
      --accent-soft: rgba(184, 92, 56, 0.14);
      --line: rgba(31, 26, 22, 0.1);
      --ok: #237750;
      --warn: #946200;
      --shadow: 0 24px 60px rgba(58, 39, 22, 0.12);
      --radius: 24px;
    }

    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(184, 92, 56, 0.16), transparent 32%),
        radial-gradient(circle at bottom right, rgba(35, 119, 80, 0.12), transparent 26%),
        linear-gradient(180deg, #f7f3eb 0%, var(--bg) 100%);
    }

    .shell {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }

    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(290px, 0.85fr);
      gap: 18px;
      align-items: stretch;
      min-height: 44vh;
    }

    .panel {
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }

    .hero-main {
      padding: 32px;
      position: relative;
      overflow: hidden;
    }

    .hero-main::after {
      content: "";
      position: absolute;
      inset: auto -12% -32% auto;
      width: 280px;
      height: 280px;
      background: radial-gradient(circle, rgba(184, 92, 56, 0.18), transparent 66%);
      pointer-events: none;
    }

    .eyebrow {
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font: 600 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--muted);
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: clamp(2.5rem, 6vw, 4.8rem);
      line-height: 0.95;
      max-width: 9ch;
      font-weight: 700;
    }

    .lede {
      margin: 18px 0 0;
      max-width: 34rem;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.7;
    }

    .hero-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 26px;
    }

    button, .button-link {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: 600 0.95rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      cursor: pointer;
      transition: transform 160ms ease, background 160ms ease, opacity 160ms ease;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    button:hover, .button-link:hover { transform: translateY(-1px); }
    button:disabled { cursor: wait; opacity: 0.7; transform: none; }
    .primary { background: var(--accent); color: #fff7f0; }
    .secondary { background: #efe5d7; color: var(--ink); }

    .hero-side {
      padding: 22px;
      display: grid;
      gap: 14px;
      align-content: start;
    }

    .stat {
      background: var(--panel-strong);
      border-radius: 18px;
      border: 1px solid var(--line);
      padding: 16px 18px;
    }

    .stat-label {
      color: var(--muted);
      font: 600 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }

    .stat-value {
      margin-top: 8px;
      font-size: 1.9rem;
      font-weight: 700;
      line-height: 1;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.25fr);
      gap: 18px;
      margin-top: 18px;
    }

    .column {
      display: grid;
      gap: 18px;
      align-content: start;
    }

    .section {
      padding: 24px;
    }

    .section h2 {
      margin: 0 0 10px;
      font-size: 1.35rem;
    }

    .section p {
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.6;
    }

    .controls {
      display: grid;
      gap: 12px;
    }

    label {
      display: grid;
      gap: 7px;
      color: var(--muted);
      font: 600 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.8);
      border-radius: 16px;
      padding: 14px 15px;
      color: var(--ink);
      font:
        500 0.98rem/1.5 ui-sans-serif,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    }

    textarea {
      min-height: 132px;
      resize: vertical;
    }

    .inline {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 132px;
      gap: 12px;
    }

    .mode-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 12px;
    }

    .mode-chip {
      padding: 10px 14px;
      border-radius: 999px;
      background: transparent;
      border: 1px solid var(--line);
      color: var(--muted);
    }

    .mode-chip.active {
      background: var(--accent-soft);
      color: var(--ink);
      border-color: rgba(184, 92, 56, 0.22);
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 999px;
      background: #f0e8da;
      color: var(--muted);
      font: 600 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--warn);
      box-shadow: 0 0 0 0 rgba(148, 98, 0, 0.35);
      animation: pulse 1.8s infinite;
    }

    .dot.ok { background: var(--ok); box-shadow: 0 0 0 0 rgba(35, 119, 80, 0.28); }

    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 currentColor; }
      70% { box-shadow: 0 0 0 12px transparent; }
      100% { box-shadow: 0 0 0 0 transparent; }
    }

    .result {
      min-height: 520px;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 16px;
    }

    .result-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      flex-wrap: wrap;
    }

    .result-head h2 {
      margin: 0;
      font-size: 1.6rem;
    }

    .result-meta {
      color: var(--muted);
      font: 600 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .answer {
      background: linear-gradient(180deg, rgba(184, 92, 56, 0.12), rgba(184, 92, 56, 0.02));
      border: 1px solid rgba(184, 92, 56, 0.18);
      border-radius: 22px;
      padding: 18px;
      white-space: pre-wrap;
      line-height: 1.65;
    }

    .empty {
      display: grid;
      place-items: center;
      text-align: center;
      border-radius: 22px;
      border: 1px dashed rgba(31, 26, 22, 0.15);
      color: var(--muted);
      min-height: 300px;
      padding: 24px;
      background: rgba(255, 255, 255, 0.38);
    }

    .list {
      display: grid;
      gap: 12px;
      align-content: start;
    }

    .item {
      padding: 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.68);
      border: 1px solid var(--line);
    }

    .item-title {
      font:
        700 0.98rem/1.35 ui-sans-serif,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    }

    .item-meta {
      margin-top: 6px;
      color: var(--muted);
      font: 600 11px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .item-body {
      margin-top: 10px;
      white-space: pre-wrap;
      line-height: 1.6;
    }

    .footer-note {
      color: var(--muted);
      font-size: 0.94rem;
      line-height: 1.6;
    }

    @media (max-width: 980px) {
      .hero, .workspace, .inline {
        grid-template-columns: 1fr;
      }
      .result { min-height: 0; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="panel hero-main">
        <div class="eyebrow">Paperless Retrieval Console</div>
        <h1>Ask your archive from the browser.</h1>
        <p class="lede">
          Run ingestion, search indexed evidence, inspect extracted facts, and ask grounded
          questions against your Paperless-backed document corpus without dropping to the terminal.
        </p>
        <div class="hero-actions">
          <button class="primary" id="refresh-health" type="button">Refresh Status</button>
          <a class="button-link secondary" href="/docs">Open API Docs</a>
        </div>
      </div>
      <aside class="panel hero-side">
        <div class="stat">
          <div class="stat-label">Service</div>
          <div class="stat-value" id="service-status">Checking</div>
        </div>
        <div class="stat">
          <div class="stat-label">Documents</div>
          <div class="stat-value" id="doc-count">--</div>
        </div>
        <div class="stat">
          <div class="stat-label">Chunks / Facts</div>
          <div class="stat-value" id="chunk-fact-count">--</div>
        </div>
        <div class="status">
          <span class="dot" id="status-dot"></span>
          <span id="status-text">Loading health</span>
        </div>
      </aside>
    </section>

    <section class="workspace">
      <div class="column">
        <section class="panel section">
          <h2>Workspace</h2>
          <p>Choose a mode, compose a request, and send it directly to the running Home LLM API.</p>
          <div class="mode-row">
            <button class="mode-chip active" data-mode="ask" type="button">Ask</button>
            <button class="mode-chip" data-mode="search" type="button">Search</button>
            <button class="mode-chip" data-mode="facts" type="button">Facts</button>
          </div>
          <form class="controls" id="query-form">
            <label>
              Prompt
              <textarea id="prompt-input" name="prompt">
What recurring debt payments do I appear to have?</textarea>
            </label>
            <div class="inline">
              <label>
                Retrieval limit
                <input id="limit-input" name="limit" type="number" min="1" max="50" value="6">
              </label>
              <label>
                Mode
                <select id="mode-input" name="mode">
                  <option value="ask">Ask</option>
                  <option value="search">Search</option>
                  <option value="facts">Facts</option>
                </select>
              </label>
            </div>
            <button class="primary" id="query-submit" type="submit">Run Request</button>
          </form>
        </section>

        <section class="panel section">
          <h2>Document Operations</h2>
          <p>
            Trigger a fresh ingest from Paperless and then review the latest extracted fact rows.
          </p>
          <div class="hero-actions">
            <button class="primary" id="ingest-button" type="button">Run Ingest</button>
            <button class="secondary" id="facts-button" type="button">Load Facts</button>
          </div>
          <p class="footer-note" id="operation-note">
            Ingest uses your current server configuration, including Paperless, PostgreSQL, Ollama,
            and optional Qdrant settings.
          </p>
        </section>
      </div>

      <div class="column">
        <section class="panel section result">
          <div class="result-head">
            <h2 id="result-title">Results</h2>
            <div class="result-meta" id="result-meta">Awaiting your first request</div>
          </div>
          <div id="answer-box" class="answer" hidden></div>
          <div id="result-body" class="empty">
            Use the workspace to ask a question, search your index, review extracted facts, or run
            ingest.
          </div>
        </section>
      </div>
    </section>
  </main>

  <script>
    const modeInput = document.getElementById("mode-input");
    const promptInput = document.getElementById("prompt-input");
    const limitInput = document.getElementById("limit-input");
    const resultTitle = document.getElementById("result-title");
    const resultMeta = document.getElementById("result-meta");
    const resultBody = document.getElementById("result-body");
    const answerBox = document.getElementById("answer-box");
    const serviceStatus = document.getElementById("service-status");
    const docCount = document.getElementById("doc-count");
    const chunkFactCount = document.getElementById("chunk-fact-count");
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");
    const operationNote = document.getElementById("operation-note");
    const chips = Array.from(document.querySelectorAll(".mode-chip"));

    function setMode(nextMode) {
      modeInput.value = nextMode;
      chips.forEach((chip) => chip.classList.toggle("active", chip.dataset.mode === nextMode));
      if (nextMode === "facts") {
        promptInput.value = "";
        promptInput.disabled = true;
        promptInput.placeholder = "Facts mode does not need a prompt.";
      } else {
        promptInput.disabled = false;
        if (!promptInput.value.trim()) {
          promptInput.value = nextMode === "ask"
            ? "What recurring debt payments do I appear to have?"
            : "mortgage";
        }
      }
      resultMeta.textContent = "Mode set to " + nextMode;
    }

    function setStatus(kind, text) {
      statusText.textContent = text;
      statusDot.classList.toggle("ok", kind === "ok");
    }

    function renderEmpty(message) {
      answerBox.hidden = true;
      answerBox.textContent = "";
      resultBody.className = "empty";
      resultBody.textContent = message;
    }

    function renderList(items, formatter) {
      if (!items.length) {
        renderEmpty("No items returned.");
        return;
      }
      resultBody.className = "list";
      resultBody.innerHTML = items.map(formatter).join("");
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, options);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || "Request failed");
      }
      return data;
    }

    async function refreshHealth() {
      try {
        const data = await fetchJson("/health");
        serviceStatus.textContent = data.status === "ok" ? "Online" : "Unknown";
        docCount.textContent = String(data.stats.documents);
        chunkFactCount.textContent = data.stats.chunks + " / " + data.stats.facts;
        setStatus("ok", "Backend healthy");
      } catch (error) {
        serviceStatus.textContent = "Error";
        docCount.textContent = "--";
        chunkFactCount.textContent = "--";
        setStatus("warn", error.message);
      }
    }

    async function runQuery(event) {
      event.preventDefault();
      const mode = modeInput.value;
      const limit = Number(limitInput.value || 6);
      const prompt = promptInput.value.trim();
      resultTitle.textContent =
        mode === "facts" ? "Facts" : mode === "search" ? "Search" : "Answer";
      resultMeta.textContent = "Working...";
      renderEmpty("Contacting Home LLM...");

      try {
        if (mode === "ask") {
          const data = await fetchJson("/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question: prompt, top_k: limit}),
          });
          answerBox.hidden = false;
          answerBox.textContent = data.answer || data.message || "No answer generated.";
          renderList(
            data.results || [],
            (item) => `
              <article class="item">
                <div class="item-title">${escapeHtml(item.file_path)}</div>
                <div class="item-meta">
                  ${escapeHtml(item.page_label)} · ${escapeHtml(item.doc_type)} ·
                  score ${Number(item.score || 0).toFixed(3)}
                </div>
                <div class="item-body">${escapeHtml(item.content)}</div>
              </article>
            `,
          );
          resultMeta.textContent = (data.results || []).length + " evidence passages";
          return;
        }

        if (mode === "search") {
          const data = await fetchJson(
            "/search?query=" +
              encodeURIComponent(prompt) +
              "&top_k=" +
              encodeURIComponent(limit)
          );
          answerBox.hidden = true;
          renderList(
            data.items || [],
            (item) => `
              <article class="item">
                <div class="item-title">${escapeHtml(item.file_path)}</div>
                <div class="item-meta">
                  ${escapeHtml(item.page_label)} · ${escapeHtml(item.doc_type)} ·
                  score ${Number(item.score || 0).toFixed(3)}
                </div>
                <div class="item-body">${escapeHtml(item.content)}</div>
              </article>
            `,
          );
          resultMeta.textContent = (data.items || []).length + " search matches";
          return;
        }

        const data = await fetchJson("/facts?limit=" + encodeURIComponent(limit));
        answerBox.hidden = true;
        renderList(
          data.items || [],
          (item) => `
            <article class="item">
              <div class="item-title">${escapeHtml(item.file_name || item.file_path)}</div>
                <div class="item-meta">
                  ${escapeHtml(item.doc_type)} · ${escapeHtml(item.fact_type)} ·
                  ${escapeHtml(item.page_label)}
                </div>
              <div class="item-body">${escapeHtml(item.fact_value)}</div>
            </article>
          `,
        );
        resultMeta.textContent = (data.items || []).length + " extracted facts";
      } catch (error) {
        answerBox.hidden = true;
        resultMeta.textContent = "Request failed";
        renderEmpty(error.message);
      }
    }

    async function runIngest() {
      operationNote.textContent = "Running ingest...";
      setStatus("warn", "Ingest in progress");
      try {
        const data = await fetchJson("/ingest", {method: "POST"});
        operationNote.textContent =
          "Indexed " + data.processed + " documents, skipped " + data.skipped + ".";
        resultTitle.textContent = "Ingest Summary";
        resultMeta.textContent =
          data.documents + " docs · " + data.chunks + " chunks · " + data.facts + " facts";
        answerBox.hidden = true;
        renderList(
          (data.errors || []).map((message, index) => ({message, index})),
          (item) => `
            <article class="item">
              <div class="item-title">Skipped item ${item.index + 1}</div>
              <div class="item-body">${escapeHtml(item.message)}</div>
            </article>
          `,
        );
        if (!(data.errors || []).length) {
          renderEmpty("Ingest completed without skipped documents.");
        }
        await refreshHealth();
        setStatus("ok", "Ingest complete");
      } catch (error) {
        operationNote.textContent = error.message;
        setStatus("warn", "Ingest failed");
      }
    }

    document.getElementById("query-form").addEventListener("submit", runQuery);
    document.getElementById("refresh-health").addEventListener("click", refreshHealth);
    document.getElementById("facts-button").addEventListener("click", () => {
      setMode("facts");
      document.getElementById("query-form").requestSubmit();
    });
    document.getElementById("ingest-button").addEventListener("click", runIngest);
    chips.forEach((chip) => chip.addEventListener("click", () => setMode(chip.dataset.mode)));

    setMode("ask");
    refreshHealth();
  </script>
</body>
</html>
"""


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

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        return APP_HTML

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

    @app.post("/ingest")
    def ingest() -> dict[str, Any]:
        config = require_config()
        try:
            return ingest_documents(config)
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
