from pathlib import Path

import pytest


def test_needs_ocr_thresholds():
    from brain.ocr import needs_ocr

    assert needs_ocr("") is True
    assert needs_ocr("   \n  ") is True
    assert needs_ocr("short") is True          # under 20 chars
    assert needs_ocr("x" * 25) is False        # real text present


def test_ocr_pdf_pages_uses_injected_backend(tmp_path):
    import pymupdf as fitz

    from brain.ocr import ocr_pdf_pages

    pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(pdf)
    doc.close()

    result = ocr_pdf_pages(pdf, pages=[1], backend=lambda img: ["hello", "world"])
    assert result == {1: "hello\nworld"}


def test_ocr_pdf_pages_defaults_to_all_pages(tmp_path):
    import pymupdf as fitz

    from brain.ocr import ocr_pdf_pages

    pdf = tmp_path / "two.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(pdf)
    doc.close()

    result = ocr_pdf_pages(pdf, backend=lambda img: ["page text"])
    assert set(result) == {0, 1}
    assert result[0] == "page text"


def test_get_ocr_backend_returns_callable():
    from brain.ocr import get_ocr_backend

    try:
        backend = get_ocr_backend()
    except ImportError:
        pytest.skip("no OCR backend installed on this machine")
    assert callable(backend)
