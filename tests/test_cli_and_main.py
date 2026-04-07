from __future__ import annotations

import argparse
import runpy

import pytest

from home_llm import cli


def test_main_returns_handler_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["home-llm", "--config", "cfg.toml", "ask", "What changed?", "--top-k", "3"],
    )
    monkeypatch.setattr(cli, "handle_ask", lambda args: 7)

    assert cli.main() == 7


def test_main_prints_value_error_and_returns_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(_: argparse.Namespace) -> int:
        raise ValueError("bad config")

    monkeypatch.setattr(
        "sys.argv",
        ["home-llm", "--config", "cfg.toml", "ingest"],
    )
    monkeypatch.setattr(cli, "handle_ingest", explode)

    assert cli.main() == 1
    assert "bad config" in capsys.readouterr().out


def test_handle_ingest_prints_summary_and_truncates_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})
    monkeypatch.setattr(
        cli,
        "ingest_documents",
        lambda config: {
            "processed": 2,
            "skipped": 12,
            "documents": 5,
            "chunks": 10,
            "facts": 8,
            "errors": [f"file-{idx}" for idx in range(12)],
        },
    )

    result = cli.handle_ingest(argparse.Namespace(config="cfg.toml"))

    assert result == 0
    output = capsys.readouterr().out
    assert "Indexed 2 documents, skipped 12." in output
    assert "Database now contains 5 documents, 10 chunks, and 8 facts." in output
    assert "- file-9" in output
    assert "- ... and 2 more" in output


def test_handle_ingest_without_error_section(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})
    monkeypatch.setattr(
        cli,
        "ingest_documents",
        lambda config: {
            "processed": 1,
            "skipped": 0,
            "documents": 1,
            "chunks": 1,
            "facts": 1,
            "errors": [],
        },
    )

    assert cli.handle_ingest(argparse.Namespace(config="cfg.toml")) == 0
    assert "Skipped files:" not in capsys.readouterr().out


def test_handle_ingest_with_short_error_list(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})
    monkeypatch.setattr(
        cli,
        "ingest_documents",
        lambda config: {
            "processed": 1,
            "skipped": 1,
            "documents": 1,
            "chunks": 1,
            "facts": 1,
            "errors": ["file-1"],
        },
    )

    assert cli.handle_ingest(argparse.Namespace(config="cfg.toml")) == 0
    output = capsys.readouterr().out
    assert "Skipped files:" in output
    assert "- file-1" in output
    assert "... and" not in output


def test_handle_ask_branches(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})

    monkeypatch.setattr(
        cli,
        "ask_question",
        lambda *_args: {"message": "nothing found", "answer": None, "results": []},
    )
    assert cli.handle_ask(argparse.Namespace(config="cfg.toml", question="q", top_k=None)) == 1
    assert "nothing found" in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "ask_question",
        lambda *_args: {"message": None, "answer": "grounded answer", "results": []},
    )
    assert cli.handle_ask(argparse.Namespace(config="cfg.toml", question="q", top_k=None)) == 0
    assert "grounded answer" in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "ask_question",
        lambda *_args: {
            "message": None,
            "answer": None,
            "results": [
                {
                    "file_path": "/tmp/a.txt",
                    "page_label": "page 1",
                    "doc_type": "statement",
                    "content": "excerpt",
                }
            ],
        },
    )
    assert cli.handle_ask(argparse.Namespace(config="cfg.toml", question="q", top_k=None)) == 0
    output = capsys.readouterr().out
    assert "[1] /tmp/a.txt (page 1, statement)" in output
    assert "excerpt" in output


def test_handle_facts_and_search(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})

    monkeypatch.setattr(cli, "list_facts", lambda *_args, **_kwargs: [])
    assert cli.handle_facts(argparse.Namespace(config="cfg.toml", limit=10)) == 1
    assert "No extracted facts found. Run ingest first." in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "list_facts",
        lambda *_args, **_kwargs: [
            {
                "file_path": "/tmp/statement.pdf",
                "doc_type": "statement",
                "fact_type": "payment_amount",
                "fact_value": "$25.00",
                "page_label": "page 1",
            }
        ],
    )
    assert cli.handle_facts(argparse.Namespace(config="cfg.toml", limit=10)) == 0
    assert "statement.pdf | statement | payment_amount = $25.00 | page 1" in capsys.readouterr().out

    monkeypatch.setattr(cli, "search_excerpts", lambda *_args, **_kwargs: [])
    assert cli.handle_search(argparse.Namespace(config="cfg.toml", query="q", top_k=3)) == 1
    assert "No search results found." in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "search_excerpts",
        lambda *_args, **_kwargs: [
            {
                "file_path": "/tmp/hit.txt",
                "page_label": "full",
                "doc_type": "document",
                "content": "matched text",
            }
        ],
    )
    assert cli.handle_search(argparse.Namespace(config="cfg.toml", query="q", top_k=3)) == 0
    assert "matched text" in capsys.readouterr().out


def test_main_module_raises_system_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("home_llm.cli.main", lambda: 4)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("home_llm.__main__", run_name="__main__")

    assert exc.value.code == 4
