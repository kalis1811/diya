from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding

from brain.config import Settings
from brain.ingest import build_documents, run_ingest

NOTE = "# Meeting notes\n\nWe decided the launch date is 2026-11-01.\n\nBudget was approved at 5000 euros."


def make_settings(tmp_path: Path) -> Settings:
    docs = tmp_path / "raw" / "docs"
    web = tmp_path / "raw" / "web"
    chats = tmp_path / "raw" / "chats"
    docs.mkdir(parents=True)
    web.mkdir(parents=True)
    chats.mkdir(parents=True)
    (docs / "meeting.md").write_text(NOTE, encoding="utf-8")
    (chats / "chat.txt").write_text(
        "15/02/26, 09:14 - Alice: the venue is the old library\n", encoding="utf-8")
    return Settings(raw_dir=tmp_path / "raw", index_path=tmp_path / "index")


class TinyEmbed(BaseEmbedding):  # md5 stand-in; must subclass BaseEmbedding (llama-index rejects duck-typing)
    def _embed(self, text: str) -> list[float]:
        import hashlib
        return [b / 255.0 for b in hashlib.md5(text.encode()).digest()[:8]]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)


def _make_pdf(path: Path, text: str):
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_build_documents_covers_docs_and_chats(tmp_path):
    settings = make_settings(tmp_path)
    docs, warnings = build_documents(settings)
    assert docs, "expected documents"
    cats = {d.metadata["category"] for d in docs}
    assert cats == {"docs", "chats"}
    assert all(len(d.metadata["file_hash"]) == 64 for d in docs)
    assert any(d.metadata.get("app") == "whatsapp" for d in docs)
    assert warnings == []


def test_pdf_routed_through_pymupdf_with_page_metadata(tmp_path):
    settings = make_settings(tmp_path)
    _make_pdf(settings.raw_dirs["docs"] / "letter.pdf", "The launch date is 2026-11-01.")
    docs, warnings = build_documents(settings)
    assert warnings == []
    pdf_docs = [d for d in docs if d.metadata["file_name"] == "letter.pdf"]
    assert pdf_docs, "pdf produced no documents"
    assert any("2026-11-01" in d.text for d in pdf_docs)
    assert all(d.metadata["page"] == 1 for d in pdf_docs)
    assert len({d.metadata["file_hash"] for d in pdf_docs}) == 1


def test_scanned_pdf_pages_go_through_ocr(tmp_path, monkeypatch):
    import brain.ocr as ocr_mod

    settings = make_settings(tmp_path)
    _make_pdf(settings.raw_dirs["docs"] / "scan.pdf", "")  # blank page -> needs_ocr
    calls = {}

    def fake_ocr(pdf_path, pages=None, backend=None, dpi=200):
        calls["pages"] = pages
        return {p: f"OCR text page {p + 1}" for p in (pages or [])}

    monkeypatch.setattr(ocr_mod, "ocr_pdf_pages", fake_ocr)
    docs, warnings = build_documents(settings)
    assert calls["pages"] == [0]
    assert any("OCR text page 1" in d.text for d in docs)
    assert warnings == []


def test_unreadable_file_warns_not_fatal(tmp_path):
    settings = make_settings(tmp_path)
    # a .docx that is not a real zip archive -> loader raises -> warning, not crash
    (settings.raw_dirs["web"] / "broken.docx").write_bytes(b"not a real docx")
    docs, warnings = build_documents(settings)
    assert docs  # the readable files still load
    assert any("broken.docx" in w for w in warnings)


def test_run_ingest_is_incremental(tmp_path):
    settings = make_settings(tmp_path)
    stats1 = run_ingest(settings, embed_model=TinyEmbed())
    assert stats1.files_new == 2
    assert stats1.chunks_indexed > 0
    stats2 = run_ingest(settings, embed_model=TinyEmbed())
    assert stats2.files_new == 0
    assert stats2.files_skipped == 2
    # modify one file -> only it re-indexes
    (settings.raw_dirs["docs"] / "meeting.md").write_text(NOTE + "\nUpdated.", encoding="utf-8")
    stats3 = run_ingest(settings, embed_model=TinyEmbed())
    assert stats3.files_new == 1
    assert stats3.files_skipped == 1

    from qdrant_client import QdrantClient
    client = QdrantClient(path=str(settings.index_path))
    try:
        points, _ = client.scroll(collection_name="brain", limit=100,
                                  with_payload=True, with_vectors=False)
        meeting_hashes = {
            p.payload["file_hash"] for p in points
            if p.payload.get("file_name") == "meeting.md"
        }
        from brain.ingest import _file_hash
        assert meeting_hashes == {_file_hash(settings.raw_dirs["docs"] / "meeting.md")}
    finally:
        client.close()


def test_run_ingest_removes_deleted_source_chunks(tmp_path):
    from qdrant_client import QdrantClient

    settings = make_settings(tmp_path)
    run_ingest(settings, embed_model=TinyEmbed())
    (settings.raw_dirs["docs"] / "meeting.md").unlink()

    stats = run_ingest(settings, embed_model=TinyEmbed())
    assert stats.files_new == 0
    assert stats.files_skipped == 1
    client = QdrantClient(path=str(settings.index_path))
    try:
        points, _ = client.scroll(collection_name="brain", limit=100,
                                  with_payload=True, with_vectors=False)
        names = {p.payload.get("file_name") for p in points}
        assert names == {"chat.txt"}
    finally:
        client.close()


def test_run_ingest_removes_chunks_when_source_is_emptied(tmp_path):
    from qdrant_client import QdrantClient

    settings = make_settings(tmp_path)
    run_ingest(settings, embed_model=TinyEmbed())
    (settings.raw_dirs["docs"] / "meeting.md").write_text("", encoding="utf-8")

    run_ingest(settings, embed_model=TinyEmbed())
    client = QdrantClient(path=str(settings.index_path))
    try:
        points, _ = client.scroll(collection_name="brain", limit=100,
                                  with_payload=True, with_vectors=False)
        names = {p.payload.get("file_name") for p in points}
        assert names == {"chat.txt"}
    finally:
        client.close()


def test_rebuild_wipes_index(tmp_path):
    settings = make_settings(tmp_path)
    run_ingest(settings, embed_model=TinyEmbed())
    stats = run_ingest(settings, rebuild=True, embed_model=TinyEmbed())
    assert stats.files_new == 2


def test_dotfiles_and_empty_documents_are_skipped(tmp_path):
    settings = make_settings(tmp_path)
    (settings.raw_dirs["docs"] / ".gitkeep").write_text("", encoding="utf-8")
    (settings.raw_dirs["web"] / ".hidden-notepad").write_text("", encoding="utf-8")
    docs, warnings = build_documents(settings)
    assert [d.metadata["file_name"] for d in docs] == ["meeting.md", "chat.txt"]
    assert warnings == []


def test_undecodable_chat_file_warns_not_fatal(tmp_path):
    settings = make_settings(tmp_path)
    (settings.raw_dirs["chats"] / "broken.txt").write_bytes(b"\xff\xfe\x00garbage")
    docs, warnings = build_documents(settings)
    assert docs  # the readable files still load
    assert any("broken.txt" in w for w in warnings)
