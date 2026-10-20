# Personal AI Brain — Phase 1 (RAG) Design Spec

- **Date:** 2026-10-18
- **Status:** Design approved by user; pending implementation plan. Amended 2026-10-18: user-directed component stack change (embeddings → qwen3-embedding:0.6b, vector store → Qdrant local, PDF via PyMuPDF, OCR, BM25 hybrid retrieval).
- **Scope:** Phase 1 = personal knowledge retrieval (RAG). Phase 2 (personal style fine-tune) gets its own spec later; roadmap summarized in §9.

## 1. Goal

A personal AI "brain" that answers questions grounded **only** in the user's own material — documents, web clippings, and chat exports — served as a local Python module and HTTP API. Everything runs on the user's machine; no data leaves it.

## 2. Environment (verified on this machine)

| Item | Value |
|---|---|
| Runtime | Windows or another supported Python platform |
| Ollama | Local Ollama service at `localhost:11434` |
| Python | 3.11 or newer, managed with `uv` |
| Models | Configurable through `BRAIN_LLM_MODEL` and `BRAIN_EMBED_MODEL` |

Model speed depends on the installed hardware. Thinking mode is off by default and can be enabled per request.

## 3. Architecture decisions

| Decision | Choice | Rationale | Rejected alternatives |
|---|---|---|---|
| LLM runtime | Ollama | Already installed; OpenAI-compatible API; automatic GPU/CPU offload | llama.cpp server, LM Studio, HF transformers |
| Generation model | `qwen3.5:4b` | Best reasoning + instruction-following runnable on 4 GB VRAM (verified); user requirement: "good at reasoning, easy to run" | `phi4-mini`, `gemma3:4b` (kept as A/B candidates); `qwen3.5:2b` (faster, weaker); `qwen3.5:9b` (too slow here) |
| Reasoning mode | Per-request `deep_think` flag, **default off**; token-capped when on | Measured: no accuracy gain on test question, 2.7× cost | Always-on thinking |
| Embeddings | `qwen3-embedding:0.6b` via Ollama | Multilingual (resolves the language-mix open item); user-directed choice; 639 MB, easy on 4 GB VRAM | `nomic-embed-text` (English-only); `bge-m3` (needs sentence-transformers) |
| Vector store | Qdrant, **embedded local mode** (`QdrantClient(path=...)`) at `data/index/` | User-directed choice; no server process; strong metadata filtering; first-class LlamaIndex integration | Chroma; LanceDB. **Caveat:** local mode allows a single process — stop the API server before re-running ingest |
| PDF reading | PyMuPDF (`fitz`) | User-directed choice; fast text extraction with per-page control; enables page-level OCR fallback | LlamaIndex default PDFReader (pypdf) |
| Scanned pages / OCR | PaddleOCR, falling back to RapidOCR (ONNX) if PaddleOCR wheels fail on Python 3.13/Windows | User-directed choice ("OCR, such as PaddleOCR"); OCR runs only on pages with no extractable text | Tesseract (extra system install) |
| Keyword search | BM25 (`rank-bm25`) fused with dense retrieval via Reciprocal Rank Fusion | User-directed choice; catches exact names, numbers, dates that embeddings miss | Dense-only retrieval |
| Reranking | **Parked** (user marked optional) | YAGNI for Phase 1; hybrid fusion covers most of the win; add a small cross-encoder later if relevance disappoints | Cross-encoder reranker now |
| RAG framework | LlamaIndex | User's explicit choice | Hand-rolled, LangChain |
| Interface | Python package + FastAPI HTTP API (port 8400) | User chose "Python/API only" | CLI chat, web UI, bot |
| Context budget | `num_ctx` 4096, `top_k` 5, chunks 512 tok / 50 overlap | Top-5 chunks + question ≈ 3K tokens, fits 4 GB VRAM budget; larger context slows the GPU-split setup | Raising `num_ctx` |

## 4. Components

```
<repository-root>/
├── pyproject.toml            # uv-managed dependencies
├── .env.example              # config template (copied to .env)
├── brain/
│   ├── __init__.py
│   ├── config.py             # env-overridable settings (below)
│   ├── ingest.py             # data/raw/** → chunks → embeddings → Qdrant (CLI: python -m brain.ingest)
│   ├── ocr.py                # scanned-page OCR: PaddleOCR first, RapidOCR (ONNX) fallback
│   ├── chat_exports.py       # chat-export parsers (WhatsApp, Telegram, +1 TBD)
│   ├── engine.py             # hybrid retrieval (dense + BM25/RRF) + grounded generation, returns answer + sources
│   ├── persona.py            # Phase-2 Step-1 persona system prompt (ships early, no training)
│   └── api.py                # FastAPI app
├── scripts/
│   └── ask.py                # one-shot CLI query for testing
├── data/
│   ├── raw/                  # user drops files: docs/  web/  chats/
│   └── index/                # Qdrant local persistence (auto-created)
├── tests/
│   ├── fixtures/             # 3 small sample files
│   └── test_smoke.py
└── docs/superpowers/specs/   # this spec
```

**`config.py` settings (all env-overridable):** `OLLAMA_HOST`, `LLM_MODEL=qwen3.5:4b`, `EMBED_MODEL=qwen3-embedding:0.6b`, `INDEX_PATH=data/index` (Qdrant local path), `RAW_DIRS={docs,web,chats}`, `CHUNK_SIZE=512`, `CHUNK_OVERLAP=50`, `TOP_K=5`, `API_PORT=8400`, `KEEP_ALIVE=30m`, `DEEP_THINK_MAX_TOKENS=1300`, `MIN_SCORE=0.25`.

**`ingest.py` behavior:** loads `data/raw/**`; `*.pdf` routed through PyMuPDF page-by-page — pages with no extractable text (`needs_ocr`) go through `ocr.py`, all other extensions via LlamaIndex readers (md/txt/docx/html); chat exports via `chat_exports.py`; `SentenceSplitter` chunking; per-document metadata `{source_path, category, file_hash}`; **skips unchanged files by hash** so re-runs are incremental; `--rebuild` deletes the Qdrant collection and re-ingests; prints per-file and total stats; unsupported/empty files skipped with a warning, never fatal.

**`ocr.py` behavior:** `needs_ocr(page_text)` → True when a page yields < 20 chars of text; `ocr_pdf_pages` lazily imports PaddleOCR and falls back to RapidOCR (ONNX, same model family, no Paddle runtime) when PaddleOCR is unavailable on this Python/OS. Slow path only — text-native pages never pay OCR cost.

**`engine.py` behavior:** hybrid retrieval — dense vector search (top `2×top_k`) + BM25 keyword search over the same chunks, fused with Reciprocal Rank Fusion (`score = Σ 1/(60+rank)`), then top `top_k` into the prompt; BM25 index built once at engine load by scrolling the Qdrant collection (derived, in-memory — not persisted separately). **Evidence gate (user flowchart): a chunk is strong evidence if its dense score ≥ `MIN_SCORE` or it has any BM25 hit; when no fused chunk is strong, generation is skipped entirely and the canned refusal is returned immediately, with the nearest sources + scores still attached.** `query(question, top_k=5, deep_think=False)` → `{answer, sources:[{file, snippet, score}], timing:{retrieve_ms, generate_ms, tokens, tok_per_s}}`. System prompt enforces: answer **only** from provided context; cite source file names; if the answer isn't in context, reply that it's not in the user's documents. Citations are structurally safe: the `sources` list is attached from retrieval, never generated by the model. `think:false` unless `deep_think=true` (then `num_predict` capped at `DEEP_THINK_MAX_TOKENS`). Persona prompt from `persona.py` prepended.

**`chat_exports.py`:** format auto-detection + parsers for WhatsApp (`.txt`: `dd/mm/yy, hh:mm - sender: message`), Telegram (`result.json` / `messages.html`), and a third app **TBD pending user sample**. Output: LlamaIndex `Document`s with metadata `{app, chat_name, date_range}`. Parsers are built against the user's real sample files, not assumptions.

**`api.py` contract:**

| Endpoint | Request | Response |
|---|---|---|
| `GET /health` | — | `{status, model, embed_model, indexed_chunks}` |
| `POST /query` | `{question, top_k?=5, deep_think?=false}` | `{answer, sources[], timing{}}` |
| `POST /query/stream` | same | SSE: token events, then final `sources` event |

Errors: **503** if Ollama is unreachable (with human-readable hint to start the Ollama app); **409** if the index is empty (hint to run ingest); **422** for malformed requests (FastAPI default).

## 5. Data flow

**Ingestion** (run when files change): `data/raw/**` → PyMuPDF/OCR + readers/parsers → `Document`s + metadata → hash-diff vs stored metadata → new/changed docs split into 512-token chunks → `qwen3-embedding:0.6b` embeddings → upsert into Qdrant (`data/index/`, local embedded mode — no server).

**Query**: question → dense search (Qdrant top `2×top_k`) + BM25 keyword search over the same chunks → RRF fusion → top-`top_k` chunks into grounded prompt (persona + rules + context) → `qwen3.5:4b` via Ollama (`think` per `deep_think`) → answer + sources + timing.

## 6. Error handling

| Condition | Behavior |
|---|---|
| Ollama not running | 503 + "start the Ollama app" message (never a raw stack trace) |
| Empty index | 409 + "run `python -m brain.ingest` first" |
| No relevant chunks (low similarity) | **No generation call** — the evidence gate short-circuits and returns the canned refusal "I don't know - that's not in your documents." with the nearest sources + scores so the caller can judge |
| Unsupported/corrupt file | Skipped during ingest with warning; ingestion continues |
| Generation runs away (deep_think) | Hard `num_predict` cap |

## 7. Testing

- **Smoke suite (`tests/test_smoke.py`, pytest):** ingest 3 fixture files into a temp index → (a) in-domain question returns the planted fact **with the fixture file cited**; (b) out-of-domain question returns the "not in your documents" behavior; (c) `/health` reports models + chunk count; (d) `/query/stream` emits token events then a sources event. Hybrid retrieval verified with unit tests: BM25 catches an exact-name/number query that dense-only ranking would bury (fake embedder); RRF fusion prefers chunks retrieved by both channels.
- **OCR tests:** `needs_ocr` thresholds on synthetic page texts; OCR backends mocked in unit tests (real OCR exercised only on a scanned fixture if PaddleOCR/RapidOCR installs cleanly).
- **Manual checks:** real-data ingestion stats; latency measurement (expect ~10–25 s per grounded answer at 13–17 tok/s; deep_think ~2–3×); exact-match queries (names/numbers) answered correctly via the BM25 channel.
- **Model A/B (optional, later):** same 10 questions against `phi4-mini` / `gemma3:4b` / `qwen3.5:2b` via config swap — empirical engine choice.

## 8. Performance expectations

Retrieval < 1 s (BM25 index built once at load; scroll time grows with corpus). Generation 150–300 tokens ≈ 10–25 s. `deep_think` ≈ 2–3× that. Ingestion speed depends on corpus size; embeddings run on GPU while the LLM is idle. OCR is the slow path: seconds per scanned page, paid only once at ingest, only on pages with no extractable text.

## 9. Phase 2 roadmap (separate spec — not built now)

1. **Persona step (no training):** extract style profile from chat exports → `persona.py` system prompt + `persona.md` style card. Ships during Phase 1.
2. **Dataset build:** `brain/finetune/prepare_data.py` — parse WhatsApp + Telegram + TBD exports (5k–50k messages), clean, thread-reconstruct, instruction pairs (context → user's reply), ~10% brain-format mix, `stats.md` for spot-checking.
3. **QLoRA training, local-only (privacy requirement):** target `qwen3.5:0.8b` (2B = stretch, likely OOM on 4 GB); Unsloth via WSL2 or PEFT+TRL native Windows; r=16/α=32, LR 2e-4, 1–3 epochs, loss on assistant tokens.
4. **Eval gate:** val loss, blind side-by-side "sounds like me" rating, RAG regression suite must still pass.
5. **Export:** merge → GGUF Q4_K_M → `ollama create brain-persona` → `LLM_MODEL=brain-persona`.

## 10. Out of scope (Phase 1)

Web UI, phone bot, multi-user/auth, cloud services, agent/tool-calling, automatic folder watching.

## 11. Open items (blocked on user)

| # | Item | Blocks |
|---|---|---|
| 1 | Name of the third chat app + one small sample export from each of the 3 apps | `chat_exports.py` parsers (Phase 1) and Phase 2 dataset |
| 2 | Real document/web-clipping locations (or drop into `data/raw/`) | Real-data ingestion |
| 3 | ~~Language mix of content~~ **Resolved by amendment:** `qwen3-embedding:0.6b` is multilingual — mixed-language content no longer changes the design | — |
