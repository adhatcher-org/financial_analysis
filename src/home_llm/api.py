from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from home_llm.config import AppConfig, load_config
from home_llm.ingest import ingest_documents
from home_llm.query_service import ask_question, get_stats, list_facts, search_excerpts

logger = logging.getLogger(__name__)

INTERNAL_SERVER_ERROR_DETAIL = "An unexpected server error occurred."

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
          <a class="button-link secondary" href="/documents">Document Management</a>
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
            Open the dedicated document workspace to run ingest jobs and watch document-level
            progress as Paperless items move through the pipeline.
          </p>
          <div class="hero-actions">
            <a class="button-link primary" href="/documents">Open Document Management</a>
            <button class="secondary" id="facts-button" type="button">View Extracted Facts</button>
          </div>
          <p class="footer-note" id="operation-note">
            The document workspace uses your current server configuration, including Paperless,
            PostgreSQL, Ollama, and optional Qdrant settings.
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

    document.getElementById("query-form").addEventListener("submit", runQuery);
    document.getElementById("refresh-health").addEventListener("click", refreshHealth);
    document.getElementById("facts-button").addEventListener("click", () => {
      setMode("facts");
      document.getElementById("query-form").requestSubmit();
    });
    chips.forEach((chip) => chip.addEventListener("click", () => setMode(chip.dataset.mode)));

    setMode("ask");
    refreshHealth();
  </script>
</body>
</html>
"""

DOCUMENTS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Document Management</title>
  <style>
    :root {
      --bg: #f2ede3;
      --panel: rgba(255, 250, 243, 0.86);
      --panel-strong: #fffaf2;
      --ink: #1f1a16;
      --muted: #6b6258;
      --accent: #125b50;
      --accent-soft: rgba(18, 91, 80, 0.14);
      --line: rgba(31, 26, 22, 0.1);
      --ok: #237750;
      --warn: #b85c38;
      --shadow: 0 24px 60px rgba(58, 39, 22, 0.12);
      --radius: 24px;
    }

    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(18, 91, 80, 0.16), transparent 30%),
        radial-gradient(circle at bottom left, rgba(184, 92, 56, 0.12), transparent 28%),
        linear-gradient(180deg, #f7f3eb 0%, var(--bg) 100%);
    }

    .shell {
      width: min(1280px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }

    .hero, .workspace {
      display: grid;
      gap: 18px;
    }

    .hero {
      grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
    }

    .workspace {
      margin-top: 18px;
      grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
    }

    .panel {
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }

    .hero-main, .hero-side, .section { padding: 28px; }
    .hero-main { position: relative; overflow: hidden; }
    .hero-main::after {
      content: "";
      position: absolute;
      right: -8%;
      bottom: -30%;
      width: 280px;
      height: 280px;
      background: radial-gradient(circle, rgba(18, 91, 80, 0.18), transparent 68%);
      pointer-events: none;
    }

    .eyebrow, .meta {
      color: var(--muted);
      font: 600 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }

    h1, h2, h3 { margin: 0; }
    h1 {
      margin-top: 14px;
      font-size: clamp(2.6rem, 6vw, 5rem);
      line-height: 0.94;
      max-width: 9ch;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.65;
    }

    .lede { margin-top: 16px; max-width: 38rem; font-size: 1.04rem; }
    .actions, .stack { display: flex; flex-wrap: wrap; gap: 12px; }
    .stack { flex-direction: column; }

    button, .button-link {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: 600 0.95rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      transition: transform 160ms ease, opacity 160ms ease;
    }
    button:hover, .button-link:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.72; cursor: wait; transform: none; }
    .primary { background: var(--accent); color: #f7fbfa; }
    .secondary { background: #ebe3d7; color: var(--ink); }

    .stats {
      display: grid;
      gap: 12px;
    }
    .stat {
      background: var(--panel-strong);
      border-radius: 18px;
      border: 1px solid var(--line);
      padding: 16px 18px;
    }
    .stat .value {
      margin-top: 8px;
      font-size: 1.9rem;
      font-weight: 700;
      line-height: 1;
    }

    .status-bar {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      margin-top: 18px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.72);
      color: var(--muted);
      font: 600 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--warn);
      animation: pulse 1.8s infinite;
    }
    .dot.ok { background: var(--ok); }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 currentColor; }
      70% { box-shadow: 0 0 0 12px transparent; }
      100% { box-shadow: 0 0 0 0 transparent; }
    }

    .grid {
      display: grid;
      gap: 18px;
      align-content: start;
    }

    .progress-shell {
      margin-top: 18px;
      background: rgba(255, 255, 255, 0.58);
      border: 1px solid var(--line);
      border-radius: 999px;
      height: 14px;
      overflow: hidden;
    }
    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), #3f8d7d);
      transition: width 240ms ease;
    }

    .section h2 { font-size: 1.4rem; margin-bottom: 10px; }
    .section p + .actions, .section p + .stack, .section p + .grid { margin-top: 16px; }

    .activity-list, .snapshot-list {
      display: grid;
      gap: 12px;
      align-content: start;
    }
    .row {
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 15px 16px;
    }
    .row-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .row-title {
      font: 700 1rem/1.35 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .row-meta {
      color: var(--muted);
      font: 600 11px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .row-body {
      margin-top: 8px;
      color: var(--muted);
      white-space: pre-wrap;
      line-height: 1.55;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 11px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--ink);
      font: 700 11px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .pill.warn { background: rgba(184, 92, 56, 0.12); }
    .pill.ok { background: rgba(35, 119, 80, 0.12); }

    .empty {
      min-height: 220px;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 24px;
      border: 1px dashed rgba(31, 26, 22, 0.15);
      border-radius: 20px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.4);
    }

    @media (max-width: 1024px) {
      .hero, .workspace { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <section class="panel hero-main">
        <div class="eyebrow">Document Management</div>
        <h1>Watch each document move through ingest.</h1>
        <p class="lede">
          Start a Paperless ingest job, follow each document as it is processed or skipped, and
          keep the running summary visible without leaving the browser.
        </p>
        <div class="actions" style="margin-top: 24px;">
          <button class="primary" id="start-job" type="button">Start Ingest Job</button>
          <button class="secondary" id="refresh-job" type="button">Refresh Job Status</button>
          <a class="button-link secondary" href="/">Back To Workspace</a>
        </div>
        <div class="status-bar">
          <span class="dot" id="job-dot"></span>
          <span id="job-status-text">Waiting for a job</span>
        </div>
        <div class="progress-shell">
          <div class="progress-bar" id="progress-bar"></div>
        </div>
      </section>
      <aside class="panel hero-side">
        <div class="stats">
          <div class="stat">
            <div class="meta">Current Job</div>
            <div class="value" id="job-id">None</div>
          </div>
          <div class="stat">
            <div class="meta">Processed / Skipped</div>
            <div class="value" id="job-counts">0 / 0</div>
          </div>
          <div class="stat">
            <div class="meta">Current Document</div>
            <div class="value" id="current-index">--</div>
            <p style="margin-top: 10px;" id="current-document">No document in flight.</p>
          </div>
        </div>
      </aside>
    </section>

    <section class="workspace">
      <section class="panel section">
        <h2>Snapshot</h2>
        <p>
          This panel shows the live state of the current job, including totals, final summary, and
          any job-level errors returned by the ingest pipeline.
        </p>
        <div class="snapshot-list" id="snapshot-list" style="margin-top: 18px;"></div>
      </section>
      <section class="panel section">
        <h2>Activity Feed</h2>
        <p>
          Each row reflects a document-level event: when processing starts, when it is stored, and
          when it is skipped because it is empty, unchanged, or errored.
        </p>
        <div class="activity-list" id="activity-list" style="margin-top: 18px;"></div>
      </section>
    </section>
  </main>

  <script>
    let currentJobId = null;
    let pollHandle = null;

    const startButton = document.getElementById("start-job");
    const refreshButton = document.getElementById("refresh-job");
    const jobStatusText = document.getElementById("job-status-text");
    const jobDot = document.getElementById("job-dot");
    const progressBar = document.getElementById("progress-bar");
    const jobId = document.getElementById("job-id");
    const jobCounts = document.getElementById("job-counts");
    const currentIndex = document.getElementById("current-index");
    const currentDocument = document.getElementById("current-document");
    const snapshotList = document.getElementById("snapshot-list");
    const activityList = document.getElementById("activity-list");

    function escapeHtml(value) {
      return String(value ?? "")
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

    function setStatus(kind, text) {
      jobStatusText.textContent = text;
      jobDot.classList.toggle("ok", kind === "ok");
    }

    function renderSnapshot(job) {
      jobId.textContent = job.job_id ? job.job_id.slice(0, 8) : "None";
      jobCounts.textContent = job.processed + " / " + job.skipped;
      currentIndex.textContent = job.total ? job.current_index + " / " + job.total : "--";
      currentDocument.textContent = job.current_document || "No document in flight.";
      const percent = job.total ? Math.round(((job.processed + job.skipped) / job.total) * 100) : 0;
      progressBar.style.width = percent + "%";
      snapshotList.innerHTML = [
        ["Status", job.status],
        ["Source", job.source || "unknown"],
        ["Total documents", String(job.total)],
        ["Processed", String(job.processed)],
        ["Skipped", String(job.skipped)],
        ["Finished at", job.finished_at || "Still running"],
        ["Error", job.error || "None"],
      ].map(([label, value]) => `
        <article class="row">
          <div class="row-head">
            <div class="row-title">${escapeHtml(label)}</div>
            <div class="row-meta">${escapeHtml(value)}</div>
          </div>
        </article>
      `).join("");

      const statusText = job.status === "completed"
        ? "Ingest complete"
        : job.status === "failed"
          ? "Ingest failed"
          : job.status === "running"
            ? "Ingest running"
            : "Waiting for a job";
      setStatus(job.status === "completed" ? "ok" : "warn", statusText);
    }

    function renderEvents(events) {
      if (!events.length) {
        activityList.innerHTML =
          '<div class="empty">Start an ingest job to watch document activity appear here.</div>';
        return;
      }
      activityList.innerHTML = events.slice().reverse().map((event) => {
        const badgeClass = event.state === "processed"
          ? "ok"
          : event.state === "skipped"
            ? "warn"
            : "";
        const reason = event.reason
          ? `<div class="row-body">${escapeHtml(event.reason)}</div>`
          : "";
        return `
          <article class="row">
            <div class="row-head">
              <div class="row-title">${escapeHtml(event.label || event.kind)}</div>
              <div class="pill ${badgeClass}">
                ${escapeHtml(event.state || event.kind)}
              </div>
            </div>
            <div class="row-meta">
              ${escapeHtml(event.source || "ingest")} ·
              ${escapeHtml(String(event.index || "--"))}/${escapeHtml(String(event.total || "--"))}
            </div>
            <div class="row-body">${escapeHtml(event.file_path || "")}</div>
            ${reason}
          </article>
        `;
      }).join("");
    }

    function updateFromSnapshot(job) {
      currentJobId = job.job_id || currentJobId;
      renderSnapshot(job);
      renderEvents(job.events || []);
      const running = job.status === "queued" || job.status === "running";
      startButton.disabled = running;
      if (running && !pollHandle && currentJobId) {
        pollHandle = window.setInterval(refreshCurrentJob, 1200);
      }
      if (!running && pollHandle) {
        window.clearInterval(pollHandle);
        pollHandle = null;
      }
    }

    async function startJob() {
      setStatus("warn", "Starting ingest job");
      const payload = await fetchJson("/ingest/jobs", {method: "POST"});
      updateFromSnapshot(payload.job);
    }

    async function refreshCurrentJob() {
      if (!currentJobId) {
        const payload = await fetchJson("/ingest/jobs/current");
        if (!payload.job) {
          renderEvents([]);
          return;
        }
        updateFromSnapshot(payload.job);
        return;
      }
      const payload = await fetchJson("/ingest/jobs/" + encodeURIComponent(currentJobId));
      updateFromSnapshot(payload.job);
    }

    startButton.addEventListener("click", () => (
      startJob().catch((error) => setStatus("warn", error.message))
    ));
    refreshButton.addEventListener("click", () => (
      refreshCurrentJob().catch((error) => setStatus("warn", error.message))
    ));
    refreshCurrentJob().catch(() => renderEvents([]));
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


@dataclass(slots=True)
class IngestJob:
    job_id: str
    status: str = "queued"
    source: str = ""
    total: int = 0
    processed: int = 0
    skipped: int = 0
    current_index: int = 0
    current_document: str = ""
    current_file_path: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "source": self.source,
            "total": self.total,
            "processed": self.processed,
            "skipped": self.skipped,
            "current_index": self.current_index,
            "current_document": self.current_document,
            "current_file_path": self.current_file_path,
            "started_at": _format_timestamp(self.started_at),
            "finished_at": _format_timestamp(self.finished_at),
            "error": self.error,
            "result": self.result,
            "events": list(self.events),
        }


class IngestJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._active_job_id: str | None = None
        self._latest_job_id: str | None = None
        self._lock = threading.Lock()

    def start_job(self) -> tuple[dict[str, Any], bool]:
        with self._lock:
            active_job = self._get_active_job_locked()
            if active_job is not None:
                return active_job.snapshot(), False

            job = IngestJob(job_id=uuid4().hex)
            self._jobs[job.job_id] = job
            self._active_job_id = job.job_id
            self._latest_job_id = job.job_id

        thread = threading.Thread(target=self._run_job, args=(job.job_id,), daemon=True)
        thread.start()
        return job.snapshot(), True

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else job.snapshot()

    def get_current_job(self) -> dict[str, Any] | None:
        with self._lock:
            job = self._get_active_job_locked()
            if job is not None:
                return job.snapshot()
            if self._latest_job_id is None:
                return None
            latest = self._jobs.get(self._latest_job_id)
            return None if latest is None else latest.snapshot()

    def _run_job(self, job_id: str) -> None:
        self._set_job_status(job_id, "running")
        try:
            result = ingest_documents(
                require_config(),
                progress_callback=lambda event: self.record(job_id, event),
            )
        except Exception as exc:  # noqa: BLE001
            self._finish_job(job_id, status="failed", error=str(exc))
            return
        self._finish_job(job_id, status="completed", result=result)

    def record(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.events.append({"timestamp": _format_timestamp(time.time()), **event})
            job.events = job.events[-120:]
            job.source = str(event.get("source", job.source))
            if event.get("kind") == "start":
                job.total = int(event.get("total", job.total))
                return
            if event.get("kind") != "document":
                return

            job.current_index = int(event.get("index", job.current_index))
            job.total = int(event.get("total", job.total))
            job.current_document = str(event.get("label", job.current_document))
            job.current_file_path = str(event.get("file_path", job.current_file_path))
            state = str(event.get("state", ""))
            if state == "processed":
                job.processed += 1
            elif state == "skipped":
                job.skipped += 1

    def _set_job_status(self, job_id: str, status: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status

    def _finish_job(
        self,
        job_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            job.finished_at = time.time()
            job.error = error
            job.result = result
            if result is not None:
                job.processed = int(result.get("processed", job.processed))
                job.skipped = int(result.get("skipped", job.skipped))
                job.total = max(job.total, job.processed + job.skipped)
            if self._active_job_id == job_id:
                self._active_job_id = None

    def _get_active_job_locked(self) -> IngestJob | None:
        if self._active_job_id is None:
            return None
        return self._jobs.get(self._active_job_id)


INGEST_JOB_MANAGER = IngestJobManager()


def _format_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))


def _internal_server_error(operation: str, exc: Exception) -> HTTPException:
    """Log diagnostic details without exposing them in an API response."""
    logger.exception("Unexpected error while %s", operation, exc_info=exc)
    return HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


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

    @app.get("/documents", response_class=HTMLResponse)
    def document_management() -> str:
        return DOCUMENTS_HTML

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
            raise _internal_server_error("answering a question", exc) from exc

    @app.post("/ingest")
    def ingest() -> dict[str, Any]:
        config = require_config()
        try:
            return ingest_documents(config)
        except Exception as exc:  # noqa: BLE001
            raise _internal_server_error("ingesting documents", exc) from exc

    @app.post("/ingest/jobs")
    def start_ingest_job() -> dict[str, Any]:
        job, started = INGEST_JOB_MANAGER.start_job()
        return {"started": started, "job": job}

    @app.get("/ingest/jobs/current")
    def current_ingest_job() -> dict[str, Any]:
        return {"job": INGEST_JOB_MANAGER.get_current_job()}

    @app.get("/ingest/jobs/{job_id}")
    def ingest_job(job_id: str) -> dict[str, Any]:
        job = INGEST_JOB_MANAGER.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Ingest job not found.")
        return {"job": job}

    @app.get("/facts")
    def facts(limit: int = Query(default=50, ge=1, le=1000)) -> dict[str, Any]:
        config = require_config()
        try:
            rows = list_facts(config, limit=limit)
            return {"items": rows, "count": len(rows)}
        except Exception as exc:  # noqa: BLE001
            raise _internal_server_error("listing facts", exc) from exc

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
            raise _internal_server_error("searching excerpts", exc) from exc

    @app.post("/search")
    def search_post(request: SearchRequest) -> dict[str, Any]:
        config = require_config()
        try:
            rows = search_excerpts(config, query=request.query, top_k=request.top_k)
            return {"items": rows, "count": len(rows)}
        except Exception as exc:  # noqa: BLE001
            raise _internal_server_error("searching excerpts", exc) from exc

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
