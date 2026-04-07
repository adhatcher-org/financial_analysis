from __future__ import annotations

import logging
from pathlib import Path

import pytest

from home_llm.document_parsers import _guess_doc_type, read_document
from home_llm.fact_extractor import _detect_summary_metric, extract_facts
from home_llm.models import DocumentChunk
from home_llm.text_utils import chunk_text, normalize_whitespace


def test_read_document_for_text_markdown_csv_and_unsupported(tmp_path: Path) -> None:
    text_path = tmp_path / "bank_statement.txt"
    text_path.write_text("hello world", encoding="utf-8")
    md_path = tmp_path / "notes.md"
    md_path.write_text("markdown body", encoding="utf-8")
    csv_path = tmp_path / "loan_data.csv"
    csv_path.write_text("a,b,\n1, 2,\n", encoding="utf-8")

    text_chunk = read_document(text_path)[0]
    md_chunk = read_document(md_path)[0]
    csv_chunk = read_document(csv_path)[0]

    assert text_chunk.doc_type == "bank"
    assert text_chunk.content == "hello world"
    assert md_chunk.page_label == "full"
    assert csv_chunk.doc_type == "loan"
    assert csv_chunk.content == "a, b\n1, 2"

    with pytest.raises(ValueError, match="Unsupported file type"):
        read_document(tmp_path / "file.docx")


def test_read_document_for_pdf_handles_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakePage:
        def __init__(self, text: str | Exception) -> None:
            self.text = text

        def extract_text(self) -> str:
            if isinstance(self.text, Exception):
                raise self.text
            return self.text

    class FakeReader:
        def __init__(self, _path: str) -> None:
            self.pages = [FakePage("First page"), FakePage(RuntimeError("boom")), FakePage("  ")]

    warnings: list[tuple[object, ...]] = []

    class FakeLogger:
        def warning(self, *args: object) -> None:
            warnings.append(args)

    pdf_path = tmp_path / "insurance_policy.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr("home_llm.document_parsers.PdfReader", FakeReader)
    real_get_logger = logging.getLogger
    monkeypatch.setattr(
        "home_llm.document_parsers.logging.getLogger",
        lambda name=None: FakeLogger() if name == "pypdf" else real_get_logger(name),
    )

    chunks = read_document(pdf_path)

    assert len(chunks) == 1
    assert chunks[0].doc_type == "insurance"
    assert chunks[0].page_label == "page 1"
    assert warnings


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mortgage.pdf", "mortgage"),
        ("loan.pdf", "loan"),
        ("insurance.pdf", "insurance"),
        ("brokerage.pdf", "investment"),
        ("investments.pdf", "investment"),
        ("retirement.pdf", "retirement"),
        ("bank.pdf", "bank"),
        ("statement.pdf", "statement"),
        ("policy.pdf", "insurance"),
        ("other.pdf", "document"),
    ],
)
def test_guess_doc_type(name: str, expected: str) -> None:
    assert _guess_doc_type(Path(name)) == expected


def test_extract_facts_and_summary_detection() -> None:
    chunk = DocumentChunk(
        file_path="/tmp/statement.pdf",
        doc_type="statement",
        page_label="page 2",
        content=(
            "Monthly payment: $123.45\n"
            "Policy number: ABCD-12345\n"
            "Account number: 1234****\n"
            "Ending balance: $1,234.56\n"
            "Due date 01/02/2026 and January 3, 2026\n"
            "Other amount USD 77.00\n"
        ),
    )

    facts = extract_facts(chunk)
    fact_types = [item["fact_type"] for item in facts]

    assert "payment_amount" in fact_types
    assert "policy_number" in fact_types
    assert "account_number" in fact_types
    assert "mentioned_amount" in fact_types
    assert fact_types.count("mentioned_date") == 2
    assert "ending_balance" in fact_types
    assert all(item["source_name"] == "statement.pdf" for item in facts)

    assert _detect_summary_metric("portfolio value: $9.00") == ("account_value", "$9.00")
    assert _detect_summary_metric("loan balance $8.00") == ("loan_balance", "$8.00")
    assert _detect_summary_metric("dwelling coverage: $7.00") == ("coverage_amount", "$7.00")
    assert _detect_summary_metric("nothing here") is None


def test_extract_facts_without_optional_collections() -> None:
    chunk = DocumentChunk(
        file_path="/tmp/blank.txt",
        doc_type="document",
        page_label="full",
        content="No structured values here.",
    )

    assert extract_facts(chunk) == []


def test_text_utils_chunking_behavior() -> None:
    assert normalize_whitespace("a   b\nc\t") == "a b c"
    assert chunk_text("   ", 10, 2) == []
    assert chunk_text("short text", 20, 5) == ["short text"]

    text = "Sentence one. Sentence two is longer. Sentence three wraps."
    chunks = chunk_text(text, chunk_size=25, chunk_overlap=5)

    assert len(chunks) >= 2
    assert all(chunk == chunk.strip() for chunk in chunks)
    assert chunk_text("abcdef ghijkl mnopqr", chunk_size=8, chunk_overlap=2) == [
        "abcdef g",
        "ghijkl",
        "l mnopqr",
    ]
