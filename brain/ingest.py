import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from brain import chat_exports
from brain.config import Settings

COLLECTION = "brain"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pdf(path: Path, category: str, needs_ocr, ocr_pdf_pages) -> tuple[list[Document], list[str]]:
    file_hash = _file_hash(path)
    doc = fitz.open(path)
    try:
        page_texts = [(i, page.get_text("text")) for i, page in enumerate(doc)]
    finally:
        doc.close()
    no_text = [i for i, text in page_texts if needs_ocr(text)]
    ocr_texts: dict[int, str] = {}
    warnings: list[str] = []
    if no_text:
        try:
            ocr_texts = ocr_pdf_pages(path, pages=no_text)
        except Exception as e:
            warnings.append(f"OCR failed for {path.name}: {e}")
    documents = []
    for i, text in page_texts:
        final = ocr_texts.get(i) or text
        if not final.strip():
            continue
        documents.append(Document(
            text=final,
            metadata={"category": category, "file_name": path.name,
                      "file_path": str(path.resolve()), "page": i + 1,
                      "file_hash": file_hash}))
    return documents, warnings


def build_documents(settings: Settings) -> tuple[list[Document], list[str]]:
    from brain import ocr as ocr_mod

    docs: list[Document] = []
    warnings: list[str] = []
    for category in ("docs", "web"):
        folder = settings.raw_dirs[category]
        if not folder.exists():
            continue
        pdfs: list[Path] = []
        others: list[Path] = []
        for f in sorted(folder.rglob("*")):
            if not f.is_file():
                continue
            (pdfs if f.suffix.lower() == ".pdf" else others).append(f)
        for pdf in pdfs:
            try:
                pdf_docs, pdf_warnings = _load_pdf(
                    pdf, category, ocr_mod.needs_ocr, ocr_mod.ocr_pdf_pages)
                docs.extend(pdf_docs)
                warnings.extend(pdf_warnings)
            except Exception as e:
                warnings.append(f"{category}: skipped {pdf.name}: {e}")
        for f in others:
            if f.name.startswith("."):
                continue
            try:
                loaded = SimpleDirectoryReader(input_files=[str(f)]).load_data()
                for d in loaded:
                    if not d.get_content().strip():
                        continue
                    d.metadata["category"] = category
                    d.metadata["file_hash"] = _file_hash(f)
                    d.metadata["file_path"] = str(f.resolve())
                    d.metadata.setdefault("file_name", f.name)
                    docs.append(d)
            except Exception as e:
                warnings.append(f"{category}: skipped {f.name}: {e}")
    chats_dir = settings.raw_dirs["chats"]
    if chats_dir.exists():
        for f in sorted(chats_dir.rglob("*")):
            if not f.is_file():
                continue
            try:
                parsed = chat_exports.detect_and_parse(f)
            except Exception as e:
                warnings.append(f"chats: skipped {f.name}: {e}")
                continue
            if parsed:
                for d in parsed:
                    d.metadata["file_path"] = str(f.resolve())
                docs.extend(parsed)
            elif f.suffix.lower() in {".txt", ".json"}:
                warnings.append(f"chats: no messages parsed from {f.name}")
    return docs, warnings


def _source_key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _existing_sources(client: QdrantClient, raw_dirs: dict[str, Path]) -> dict[str, list[tuple[str, str]]]:
    """Map normalized source paths to indexed (content hash, point id) pairs."""
    sources: dict[str, list[tuple[str, str]]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION, limit=256, offset=offset,
            with_payload=True, with_vectors=False)
        for p in points:
            payload = p.payload or {}
            h = payload.get("file_hash")
            if not h:
                continue
            path = payload.get("file_path")
            if not path:
                # Older chat chunks did not store file_path. Recover the
                # common non-nested case from their category and file name.
                category, name = payload.get("category"), payload.get("file_name")
                if category in raw_dirs and name:
                    path = raw_dirs[category] / name
            if path:
                sources.setdefault(_source_key(path), []).append((h, str(p.id)))
        if offset is None:
            return sources


def _current_source_paths(raw_dirs: dict[str, Path]) -> set[str]:
    return {
        _source_key(path)
        for folder in raw_dirs.values() if folder.exists()
        for path in folder.rglob("*") if path.is_file()
    }


def _current_source_hashes(raw_dirs: dict[str, Path]) -> dict[str, str]:
    hashes = {}
    for folder in raw_dirs.values():
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                try:
                    hashes[_source_key(path)] = _file_hash(path)
                except OSError:
                    # Keep its old chunks if the file cannot currently be read.
                    continue
    return hashes


def run_ingest(settings: Settings, rebuild: bool = False, embed_model=None) -> "IngestStats":
    if rebuild and settings.index_path.exists():
        # Qdrant local mode: delete_collection in-process leaves stale persisted
        # segments behind (observed: duplicate + empty points after --rebuild).
        # Wiping the directory is the only reliable wipe; it also fails loudly
        # (OS error) if another process holds the index open.
        shutil.rmtree(settings.index_path)
    client = QdrantClient(path=str(settings.index_path))
    try:
        all_docs, warnings = build_documents(settings)
        existing = (_existing_sources(client, settings.raw_dirs)
                    if client.collection_exists(COLLECTION) else {})
        existing_keys = {
            (path, file_hash)
            for path, records in existing.items()
            for file_hash, _point_id in records
        }
        new_docs = [
            d for d in all_docs
            if d.metadata.get("file_hash") and d.metadata.get("file_path") and
            (_source_key(d.metadata["file_path"]), d.metadata["file_hash"]) not in existing_keys
        ]
        indexed_source_keys = {
            (_source_key(d.metadata["file_path"]), d.metadata["file_hash"])
            for d in all_docs if d.metadata.get("file_path") and d.metadata.get("file_hash")
        }
        new_source_keys = {
            (_source_key(d.metadata["file_path"]), d.metadata["file_hash"])
            for d in new_docs
        }
        current_paths = _current_source_paths(settings.raw_dirs)
        current_hashes = _current_source_hashes(settings.raw_dirs)
        hashes_by_path: dict[str, set[str]] = {}
        for path, file_hash in indexed_source_keys:
            hashes_by_path.setdefault(path, set()).add(file_hash)

        def has_read_warning(path: str) -> bool:
            name = Path(path).name
            return any(
                f"skipped {name}:" in warning or
                f"OCR failed for {name}:" in warning or
                f"no messages parsed from {name}" in warning
                for warning in warnings
            )

        stale_ids = [
            point_id
            for path, records in existing.items()
            for file_hash, point_id in records
            if path not in current_paths or
            (path in hashes_by_path and file_hash not in hashes_by_path[path]) or
            (path in current_hashes and current_hashes[path] != file_hash and
             path not in hashes_by_path and not has_read_warning(path))
        ]

        if new_docs:
            embed = embed_model or OllamaEmbedding(model_name=settings.embed_model,
                                                   base_url=settings.ollama_host)
            splitter = SentenceSplitter(chunk_size=settings.chunk_size,
                                        chunk_overlap=settings.chunk_overlap)
            # Local Qdrant does not support payload indexes. LlamaIndex creates
            # a document-ID index by default, which only adds a warning here;
            # the app does not filter queries by that internal ID.
            vector_store = QdrantVectorStore(
                client=client, collection_name=COLLECTION, index_doc_id=False)
            storage = StorageContext.from_defaults(vector_store=vector_store)
            # Add the replacement version first. If embedding fails, old chunks
            # remain available; stale versions are removed only after success.
            VectorStoreIndex.from_documents(
                new_docs, storage_context=storage, embed_model=embed,
                transformations=[splitter], show_progress=False)
        if stale_ids:
            client.delete(collection_name=COLLECTION,
                          points_selector=models.PointIdsList(points=stale_ids))
        chunks = client.count(COLLECTION, exact=True).count if client.collection_exists(COLLECTION) else 0
        new_file_count = len({path for path, _file_hash in new_source_keys})
        skipped_file_count = len({
            _source_key(d.metadata["file_path"])
            for d in all_docs
            if d.metadata.get("file_path") and d.metadata.get("file_hash") and
            (_source_key(d.metadata["file_path"]), d.metadata["file_hash"]) in existing_keys
        })
    finally:
        client.close()
    return IngestStats(files_new=new_file_count, files_skipped=skipped_file_count,
                       chunks_indexed=chunks, warnings=warnings)


@dataclass
class IngestStats:
    files_new: int
    files_skipped: int
    chunks_indexed: int
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        out = f"new files: {self.files_new}, skipped: {self.files_skipped}, chunks indexed: {self.chunks_indexed}"
        if self.warnings:
            out += "\nwarnings:\n" + "\n".join(f"  - {w}" for w in self.warnings)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest data/raw into the brain index")
    ap.add_argument("--rebuild", action="store_true", help="wipe the index and re-ingest everything")
    args = ap.parse_args()
    from brain.config import get_settings
    print(run_ingest(get_settings(), rebuild=args.rebuild))


if __name__ == "__main__":
    main()
