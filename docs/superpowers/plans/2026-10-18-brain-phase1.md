# Personal AI Brain — Phase 1 (RAG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local personal-knowledge RAG service: ingest the user's documents (PDF via PyMuPDF with OCR for scanned pages), web clippings, and chat exports into a Qdrant index, answer questions grounded only in that material using hybrid retrieval (dense + BM25 fused by RRF) with cited sources, via a Python module and FastAPI API.

**Architecture:** LlamaIndex orchestrates ingestion (readers + PyMuPDF page extraction + OCR fallback → SentenceSplitter chunks) and dense retrieval (Qdrant local embedded mode, `qwen3-embedding:0.6b` embeddings). The engine fuses dense ranking with BM25 keyword ranking (Reciprocal Rank Fusion) so exact names/numbers/dates surface. Generation calls Ollama's `/api/chat` directly through a thin `httpx` client (exact control of `think`, `num_predict`, `keep_alive`). FastAPI exposes `/health`, `/query`, `/query/stream`.

**Tech Stack:** Python ≥3.11, uv, llama-index-core, llama-index-embeddings-ollama, llama-index-vector-stores-qdrant, qdrant-client, pymupdf, rank-bm25, numpy, paddleocr OR rapidocr-onnxruntime (OCR, decided in Task 4), fastapi, uvicorn, httpx, pytest. Ollama v0.35.1 with `qwen3.5:4b` + `qwen3-embedding:0.6b`.

**Spec:** `docs/superpowers/specs/2026-10-18-personal-ai-brain-design.md` (read it first — this plan argues from it).

## Global Constraints

- Python `>=3.11`; use `uv` exclusively (never bare `pip`); run everything from the repository root.
- All inference is local via Ollama at `http://localhost:11434`. Models are exactly `qwen3.5:4b` (generation) and `qwen3-embedding:0.6b` (embeddings) unless overridden by `BRAIN_*` env vars.
- Fixed defaults (spec §3): API port `8400`, chunk `512` tokens / overlap `50`, `top_k` `5`, `deep_think` default OFF with `num_predict` cap `1300`, `keep_alive` `30m`, `min_score` `0.25`.
- **Always pass `embed_model=` explicitly** to every LlamaIndex index/embed call — the global default would try OpenAI and fail.
- `.env` and `data/` are gitignored (already in `.gitignore` at repo root). Never commit user data.
- Every task: write failing test → verify fail → implement → verify pass → commit. Commit message style: `feat:`/`test:`/`chore:` prefixes.
- **Qdrant runs in local embedded mode** (`QdrantClient(path=...)`): only ONE process may open `data/index` at a time. Stop the API server before running ingest. This is a documented caveat, not a bug to work around.
- **OCR backends are lazy imports** (`brain/ocr.py`); unit tests ALWAYS inject a fake backend via the `backend=` parameter — tests never depend on PaddleOCR/RapidOCR being installed.
- Reranking is out of scope (user parked it).

---

### Task 1: Project scaffold, dependencies, config, persona

**Files:**
- Create: `pyproject.toml`, `brain/__init__.py`, `brain/config.py`, `brain/persona.py`, `.env.example`, `data/raw/docs/.gitkeep`, `data/raw/web/.gitkeep`, `data/raw/chats/.gitkeep`
- Test: `tests/test_config.py`, `tests/conftest.py`

**Interfaces:**
- Produces: `brain.config.Settings` (frozen dataclass: `ollama_host: str, llm_model: str, embed_model: str, index_path: Path, raw_dir: Path, chunk_size: int, chunk_overlap: int, top_k: int, api_port: int, keep_alive: str, deep_think_max_tokens: int, min_score: float`, plus property `raw_dirs -> dict[str, Path]`), `brain.config.get_settings() -> Settings` (cached, env-overridable), `brain.persona.build_system_prompt(context_block: str) -> str`, `brain.persona.REFUSAL` (the exact refusal sentence, consumed by the engine's evidence gate in Task 6).

- [ ] **Step 1: Create pyproject.toml and package skeleton**

`pyproject.toml`:

```toml
[project]
name = "brain"
version = "0.1.0"
description = "Personal AI brain - local RAG over your own material"
requires-python = ">=3.11"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["brain"]

[dependency-groups]
dev = ["pytest>=8.0"]
```

Then:

```bash
mkdir -p brain tests scripts data/raw/docs data/raw/web data/raw/chats
touch brain/__init__.py data/raw/docs/.gitkeep data/raw/web/.gitkeep data/raw/chats/.gitkeep
uv add llama-index-core llama-index-embeddings-ollama llama-index-vector-stores-qdrant qdrant-client fastapi uvicorn httpx pymupdf rank-bm25 python-docx beautifulsoup4
uv sync
```

Verify imports and that Qdrant local mode actually works on this Python:

```bash
uv run python -c "import qdrant_client, llama_index.core, fitz, rank_bm25, fastapi, httpx; print('imports OK')"
uv run python -c "
import tempfile
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
d = tempfile.mkdtemp()
c = QdrantClient(path=d)
c.create_collection('t', vectors_config=VectorParams(size=4, distance=Distance.COSINE))
print('qdrant local OK, points:', c.count('t', exact=True).count)
c.close()
"
```

Expected: `imports OK` and `qdrant local OK, points: 0`. **If either fails, STOP and report** — do not improvise a different vector store.

- [ ] **Step 2: Write failing config test**

`tests/test_config.py`:

```python
import dataclasses
from pathlib import Path

from brain.config import Settings, get_settings


def test_defaults_match_spec():
    s = Settings()
    assert s.ollama_host == "http://localhost:11434"
    assert s.llm_model == "qwen3.5:4b"
    assert s.embed_model == "qwen3-embedding:0.6b"
    assert s.chunk_size == 512
    assert s.chunk_overlap == 50
    assert s.top_k == 5
    assert s.api_port == 8400
    assert s.keep_alive == "30m"
    assert s.deep_think_max_tokens == 1300
    assert s.min_score == 0.25


def test_raw_dirs_categories():
    s = Settings(raw_dir=Path("C:/tmp/x"))
    assert set(s.raw_dirs) == {"docs", "web", "chats"}
    assert s.raw_dirs["docs"] == Path("C:/tmp/x") / "docs"


def test_settings_are_frozen():
    s = Settings()
    try:
        s.top_k = 9
        raise AssertionError("Settings should be frozen")
    except dataclasses.FrozenInstanceError:
        pass


def test_env_override(monkeypatch):
    monkeypatch.setenv("BRAIN_LLM_MODEL", "brain-persona")
    monkeypatch.setenv("BRAIN_API_PORT", "9999")
    monkeypatch.setenv("BRAIN_INDEX_PATH", "D:/elsewhere/idx")
    get_settings.cache_clear()
    s = get_settings()
    assert s.llm_model == "brain-persona"
    assert s.api_port == 9999
    assert s.index_path == Path("D:/elsewhere/idx")
    get_settings.cache_clear()  # don't leak into other tests
```

Run: `uv run pytest tests/test_config.py -v` — Expected: FAIL (`ModuleNotFoundError: brain.config`).

- [ ] **Step 3: Implement config.py and persona.py**

`brain/config.py`:

```python
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DEFAULTS = {
    "BRAIN_OLLAMA_HOST": "http://localhost:11434",
    "BRAIN_LLM_MODEL": "qwen3.5:4b",
    "BRAIN_EMBED_MODEL": "qwen3-embedding:0.6b",
    "BRAIN_CHUNK_SIZE": "512",
    "BRAIN_CHUNK_OVERLAP": "50",
    "BRAIN_TOP_K": "5",
    "BRAIN_API_PORT": "8400",
    "BRAIN_KEEP_ALIVE": "30m",
    "BRAIN_DEEP_THINK_MAX_TOKENS": "1300",
    "BRAIN_MIN_SCORE": "0.25",
}


def _load_dotenv() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    ollama_host: str = _DEFAULTS["BRAIN_OLLAMA_HOST"]
    llm_model: str = _DEFAULTS["BRAIN_LLM_MODEL"]
    embed_model: str = _DEFAULTS["BRAIN_EMBED_MODEL"]
    index_path: Path = PROJECT_ROOT / "data" / "index"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    chunk_size: int = int(_DEFAULTS["BRAIN_CHUNK_SIZE"])
    chunk_overlap: int = int(_DEFAULTS["BRAIN_CHUNK_OVERLAP"])
    top_k: int = int(_DEFAULTS["BRAIN_TOP_K"])
    api_port: int = int(_DEFAULTS["BRAIN_API_PORT"])
    keep_alive: str = _DEFAULTS["BRAIN_KEEP_ALIVE"]
    deep_think_max_tokens: int = int(_DEFAULTS["BRAIN_DEEP_THINK_MAX_TOKENS"])
    min_score: float = float(_DEFAULTS["BRAIN_MIN_SCORE"])

    @property
    def raw_dirs(self) -> dict[str, Path]:
        return {cat: self.raw_dir / cat for cat in ("docs", "web", "chats")}


@lru_cache
def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        ollama_host=os.environ.get("BRAIN_OLLAMA_HOST", _DEFAULTS["BRAIN_OLLAMA_HOST"]),
        llm_model=os.environ.get("BRAIN_LLM_MODEL", _DEFAULTS["BRAIN_LLM_MODEL"]),
        embed_model=os.environ.get("BRAIN_EMBED_MODEL", _DEFAULTS["BRAIN_EMBED_MODEL"]),
        index_path=Path(os.environ.get("BRAIN_INDEX_PATH", str(PROJECT_ROOT / "data" / "index"))),
        raw_dir=Path(os.environ.get("BRAIN_RAW_DIR", str(PROJECT_ROOT / "data" / "raw"))),
        chunk_size=int(os.environ.get("BRAIN_CHUNK_SIZE", _DEFAULTS["BRAIN_CHUNK_SIZE"])),
        chunk_overlap=int(os.environ.get("BRAIN_CHUNK_OVERLAP", _DEFAULTS["BRAIN_CHUNK_OVERLAP"])),
        top_k=int(os.environ.get("BRAIN_TOP_K", _DEFAULTS["BRAIN_TOP_K"])),
        api_port=int(os.environ.get("BRAIN_API_PORT", _DEFAULTS["BRAIN_API_PORT"])),
        keep_alive=os.environ.get("BRAIN_KEEP_ALIVE", _DEFAULTS["BRAIN_KEEP_ALIVE"]),
        deep_think_max_tokens=int(os.environ.get("BRAIN_DEEP_THINK_MAX_TOKENS", _DEFAULTS["BRAIN_DEEP_THINK_MAX_TOKENS"])),
        min_score=float(os.environ.get("BRAIN_MIN_SCORE", _DEFAULTS["BRAIN_MIN_SCORE"])),
    )
```

`brain/persona.py` (placeholder persona — Phase 2 Step 1 refines it from real chat exports; the RAG rules are load-bearing now):

```python
PERSONA_PROMPT = (
    "You are the user's personal AI brain. You answer questions about the user's own "
    "life and material: their documents, saved articles, and chat history. "
    "You are precise, down-to-earth, and never invent facts."
)

REFUSAL = "I don't know - that's not in your documents."

RAG_RULES = (
    "STRICT RULES:\n"
    "1. Answer ONLY using the CONTEXT provided below. Never use outside knowledge.\n"
    f"2. If the CONTEXT does not contain the answer, reply exactly: \"{REFUSAL}\"\n"
    "3. Cite the source file names you used, in square brackets, e.g. [notes.md].\n"
    "4. Be concise. Answer the question directly, then stop."
)


def build_system_prompt(context_block: str) -> str:
    return f"{PERSONA_PROMPT}\n\n{RAG_RULES}\n\nCONTEXT:\n{context_block}"
```

`.env.example`:

```
# Copy to .env to override. Every var is optional.
# BRAIN_OLLAMA_HOST=http://localhost:11434
# BRAIN_LLM_MODEL=qwen3.5:4b
# BRAIN_EMBED_MODEL=qwen3-embedding:0.6b
# BRAIN_INDEX_PATH=data/index
# BRAIN_API_PORT=8400
# BRAIN_TOP_K=5
# BRAIN_KEEP_ALIVE=30m
```

Also create `tests/conftest.py` (used by later tasks; empty for now):

```python
```

- [ ] **Step 4: Run config tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock brain/ tests/ data/raw/ .env.example
git commit -m "feat: scaffold brain package with config, persona, deps"
```

---

### Task 2: Ollama HTTP client

**Files:**
- Create: `brain/ollama_client.py`
- Test: `tests/test_ollama_client.py`

**Interfaces:**
- Produces: `OllamaClient(base_url: str, model: str, timeout: float = 300.0, transport: httpx.BaseTransport | None = None)` with:
  - `is_up() -> bool`
  - `chat(messages: list[dict], *, think: bool = False, max_tokens: int | None = None, keep_alive: str = "30m") -> ChatResult` where `ChatResult(text: str, thinking: str, eval_count: int, tok_per_s: float)`
  - `chat_stream(messages: list[dict], *, think: bool = False, max_tokens: int | None = None, keep_alive: str = "30m") -> Iterator[str]` (yields content deltas only)
  - `OllamaDown(OllamaError)` raised when the server is unreachable.
- Consumes: nothing from other tasks (uses only httpx).

- [ ] **Step 1: Write failing tests**

`tests/test_ollama_client.py`:

```python
import json

import httpx
import pytest

from brain.ollama_client import OllamaClient, OllamaDown


def make_client(handler) -> OllamaClient:
    return OllamaClient("http://fake", "qwen3.5:4b", transport=httpx.MockTransport(handler))


def test_chat_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "qwen3.5:4b"
        assert body["stream"] is False
        assert body["think"] is False
        assert body["keep_alive"] == "30m"
        assert "options" not in body
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "The answer is 3."},
            "thinking": "hmm let me think",
            "eval_count": 42,
            "eval_duration": 2_000_000_000,
        })

    c = make_client(handler)
    res = c.chat([{"role": "user", "content": "q"}])
    assert res.text == "The answer is 3."
    assert res.thinking == "hmm let me think"
    assert res.eval_count == 42
    assert res.tok_per_s == 21.0  # 42 tokens / 2s


def test_chat_sends_think_and_num_predict_when_asked():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}, "eval_count": 1, "eval_duration": 1})

    c = make_client(handler)
    c.chat([{"role": "user", "content": "q"}], think=True, max_tokens=1300, keep_alive="5m")
    assert captured["think"] is True
    assert captured["options"] == {"num_predict": 1300}
    assert captured["keep_alive"] == "5m"


def test_unreachable_raises_ollama_down():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no server")

    c = make_client(handler)
    with pytest.raises(OllamaDown):
        c.chat([{"role": "user", "content": "q"}])


def test_is_up_true_and_false():
    up = make_client(lambda r: httpx.Response(200, json={"version": "0.35.1"}))
    down = make_client(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x")))
    assert up.is_up() is True
    assert down.is_up() is False


def test_chat_stream_yields_content_deltas():
    lines = "\n".join([
        json.dumps({"message": {"content": "Hel"}, "done": False}),
        json.dumps({"message": {"content": "lo"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]) + "\n"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(200, content=lines.encode())

    c = make_client(handler)
    assert list(c.chat_stream([{"role": "user", "content": "q"}])) == ["Hel", "lo"]
```

Run: `uv run pytest tests/test_ollama_client.py -v` — Expected: FAIL (module missing).

- [ ] **Step 2: Implement**

`brain/ollama_client.py`:

```python
import json
from collections.abc import Iterator
from dataclasses import dataclass

import httpx


class OllamaError(RuntimeError):
    pass


class OllamaDown(OllamaError):
    pass


@dataclass
class ChatResult:
    text: str
    thinking: str
    eval_count: int
    tok_per_s: float


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0,
                 transport: httpx.BaseTransport | None = None):
        self._base = base_url.rstrip("/")
        self.model = model
        self._timeout = timeout
        self._transport = transport

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(timeout=timeout, transport=self._transport)

    def is_up(self) -> bool:
        try:
            with self._client(3.0) as c:
                return c.get(f"{self._base}/api/version").status_code == 200
        except httpx.HTTPError:
            return False

    def _payload(self, messages: list[dict], *, think: bool, max_tokens: int | None,
                 keep_alive: str, stream: bool) -> dict:
        payload = {"model": self.model, "messages": messages, "stream": stream,
                   "think": think, "keep_alive": keep_alive}
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}
        return payload

    def chat(self, messages: list[dict], *, think: bool = False,
             max_tokens: int | None = None, keep_alive: str = "30m") -> ChatResult:
        payload = self._payload(messages, think=think, max_tokens=max_tokens,
                                keep_alive=keep_alive, stream=False)
        try:
            with self._client(self._timeout) as c:
                r = c.post(f"{self._base}/api/chat", json=payload)
        except httpx.HTTPError as e:
            raise OllamaDown(f"Ollama unreachable at {self._base}: {e}") from e
        if r.status_code != 200:
            raise OllamaError(f"Ollama returned {r.status_code}: {r.text[:200]}")
        d = r.json()
        msg = d.get("message", {})
        count = d.get("eval_count", 0)
        dur = d.get("eval_duration", 0)
        return ChatResult(
            text=msg.get("content", ""),
            thinking=d.get("thinking", "") or msg.get("thinking", "") or "",
            eval_count=count,
            tok_per_s=round(count / (dur / 1e9), 1) if dur else 0.0,
        )

    def chat_stream(self, messages: list[dict], *, think: bool = False,
                    max_tokens: int | None = None,
                    keep_alive: str = "30m") -> Iterator[str]:
        payload = self._payload(messages, think=think, max_tokens=max_tokens,
                                keep_alive=keep_alive, stream=True)
        try:
            with self._client(self._timeout) as c:
                with c.stream("POST", f"{self._base}/api/chat", json=payload) as r:
                    if r.status_code != 200:
                        raise OllamaError(f"Ollama returned {r.status_code}")
                    for line in r.iter_lines():
                        if not line:
                            continue
                        d = json.loads(line)
                        delta = d.get("message", {}).get("content", "")
                        if delta:
                            yield delta
                        if d.get("done"):
                            break
        except httpx.HTTPError as e:
            raise OllamaDown(f"Ollama unreachable at {self._base}: {e}") from e
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_ollama_client.py -v`
Expected: 5 PASS.

- [ ] **Step 4: Commit**

```bash
git add brain/ollama_client.py tests/test_ollama_client.py
git commit -m "feat: Ollama HTTP client with think/num_predict/keep_alive control"
```

---

### Task 3: Chat-export parsers (WhatsApp, Telegram)

**Files:**
- Create: `brain/chat_exports.py`
- Test: `tests/test_chat_exports.py`, fixtures `tests/fixtures/chat_sample_android.txt`, `tests/fixtures/chat_sample_ios.txt`, `tests/fixtures/telegram_result.json`

**Interfaces:**
- Produces: `ParsedMessage(sender: str, text: str, timestamp: str)`, `parse_whatsapp(path: Path) -> list[ParsedMessage]`, `parse_telegram_json(path: Path) -> list[ParsedMessage]`, `detect_and_parse(path: Path) -> list[Document]` (llama_index `Document`s with metadata `{"category": "chats", "app": str, "chat_name": str, "date_range": str, "file_name": str, "file_hash": str}` and text of `sender: message` lines).
- Note: parsers are built against synthetic samples now; **they must be re-verified against the user's real exports** (spec open item) before real ingestion.

- [ ] **Step 1: Create fixtures**

`tests/fixtures/chat_sample_android.txt`:

```
15/02/26, 09:14 - Alice: hey are you coming tomorrow?
15/02/26, 09:15 - Bob: yeah probably
depends on when it starts tbh
15/02/26, 09:16 - Alice: cool, it's at 10
<Media omitted>
15/02/26, 09:20 - Bob: ok see you there 👍
```

`tests/fixtures/chat_sample_ios.txt`:

```
[15/02/26, 09:14:00] Alice: project update sent
[15/02/26, 09:15:30] Bob: got it, reviewing now
```

`tests/fixtures/telegram_result.json`:

```json
{
  "name": "Project Chat",
  "messages": [
    {"type": "service", "date": "2026-02-15T09:00:00", "actor": "Bob", "action": "joined group"},
    {"type": "message", "date": "2026-02-15T09:14:00", "from": "Alice", "text": "did you finish the report?"},
    {"type": "message", "date": "2026-02-15T09:20:00", "from": "Bob", "text": ["yes ", {"type": "link", "text": "here"}, " done"]}
  ]
}
```

- [ ] **Step 2: Write failing tests**

`tests/test_chat_exports.py`:

```python
from pathlib import Path

from brain.chat_exports import detect_and_parse, parse_telegram_json, parse_whatsapp

FIX = Path(__file__).parent / "fixtures"


def test_parse_whatsapp_android_multiline_and_media():
    msgs = parse_whatsapp(FIX / "chat_sample_android.txt")
    assert len(msgs) == 4
    assert msgs[0].sender == "Alice"
    assert msgs[0].text == "hey are you coming tomorrow?"
    assert msgs[1].sender == "Bob"
    assert "depends on when it starts tbh" in msgs[1].text
    assert "<Media omitted>" in msgs[2].text
    assert msgs[3].sender == "Bob"
    assert msgs[3].text == "ok see you there 👍"


def test_parse_whatsapp_ios_bracket_format():
    msgs = parse_whatsapp(FIX / "chat_sample_ios.txt")
    assert len(msgs) == 2
    assert msgs[1].sender == "Bob"
    assert msgs[1].text == "got it, reviewing now"


def test_parse_telegram_json_flattens_rich_text():
    msgs = parse_telegram_json(FIX / "telegram_result.json")
    assert len(msgs) == 2  # service messages skipped
    assert msgs[0].sender == "Alice"
    assert msgs[1].text == "yes here done"


def test_detect_and_parse_builds_documents_with_metadata():
    docs = detect_and_parse(FIX / "telegram_result.json")
    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["category"] == "chats"
    assert meta["app"] == "telegram"
    assert meta["chat_name"] == "Project Chat"
    assert meta["file_name"] == "telegram_result.json"
    assert len(meta["file_hash"]) == 64  # sha256 hex
    assert "Alice: did you finish the report?" in docs[0].text
```

Run: `uv run pytest tests/test_chat_exports.py -v` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

`brain/chat_exports.py`:

```python
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document


@dataclass
class ParsedMessage:
    sender: str
    text: str
    timestamp: str


# Matches both Android ("15/02/26, 09:14 - Name: msg") and iOS ("[15/02/26, 09:14:00] Name: msg")
_WA_LINE = re.compile(
    r"^(?:\[(?P<bdt>[^\]]+)\]\s*|(?P<adt>\d{1,2}[./]\d{1,2}[./]\d{2,4},?\s+\d{1,2}:\d{2}"
    r"(?::\d{2})?(?:\s*[APap][Mm])?)\s*-\s*)"
    r"(?P<sender>[^:]{1,80}?):\s?(?P<text>.*)$"
)


def parse_whatsapp(path: Path) -> list[ParsedMessage]:
    messages: list[ParsedMessage] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _WA_LINE.match(line.strip())
        if m:
            ts = (m.group("bdt") or m.group("adt") or "").strip()
            messages.append(ParsedMessage(m.group("sender").strip(), m.group("text").strip(), ts))
        elif messages and line.strip():
            messages[-1].text += "\n" + line.strip()  # multiline continuation
    return messages


def _flatten_text(text_field) -> str:
    if isinstance(text_field, str):
        return text_field
    parts = []
    for part in text_field or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            parts.append(str(part.get("text", "")))
    return "".join(parts)


def parse_telegram_json(path: Path) -> list[ParsedMessage]:
    data = json.loads(path.read_text(encoding="utf-8"))
    messages = []
    for m in data.get("messages", []):
        if m.get("type") != "message":
            continue
        text = _flatten_text(m.get("text")).strip()
        if not text:
            continue
        messages.append(ParsedMessage(m.get("from") or m.get("actor") or "unknown", text, m.get("date", "")))
    return messages


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detect_and_parse(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            msgs = parse_telegram_json(path)
        except (json.JSONDecodeError, KeyError):
            return []
        app, chat_name = "telegram", str(data.get("name") or path.stem)
    elif suffix == ".txt":
        msgs, app, chat_name = parse_whatsapp(path), "whatsapp", path.stem
    else:
        return []
    if not msgs:
        return []
    body = "\n".join(f"{m.sender}: {m.text}" for m in msgs)
    dates = sorted({m.timestamp for m in msgs if m.timestamp})
    date_range = f"{dates[0]} .. {dates[-1]}" if dates else "unknown"
    return [Document(
        text=body,
        metadata={"category": "chats", "app": app, "chat_name": chat_name,
                  "date_range": date_range,
                  "file_name": path.name, "file_hash": _file_hash(path)},
    )]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_chat_exports.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add brain/chat_exports.py tests/test_chat_exports.py tests/fixtures/
git commit -m "feat: WhatsApp and Telegram chat export parsers"
```

---

### Task 4: OCR module for scanned PDF pages

**Files:**
- Create: `brain/ocr.py`
- Test: `tests/test_ocr.py`

**Interfaces:**
- Produces:
  - `needs_ocr(page_text: str, min_chars: int = 20) -> bool` — True when a page yields less than `min_chars` of extractable text.
  - `get_ocr_backend() -> Callable[[Any], list[str]]` — lazily loads PaddleOCR; falls back to RapidOCR (ONNX) if PaddleOCR is not importable. Returns a callable `numpy_image -> list[str]` (recognized text lines). Backend instance is cached at module level.
  - `ocr_pdf_pages(path: Path, pages: list[int] | None = None, backend=None, dpi: int = 200) -> dict[int, str]` — renders the given 0-based page numbers (default: all) via PyMuPDF and OCRs each; returns `{page_number: ocr_text}`. `backend` parameter is the test seam — when given, no OCR library is imported.
- Consumes: nothing from other tasks (uses pymupdf + numpy + one OCR library).

- [ ] **Step 1: Add OCR dependencies (with documented fallback decision)**

```bash
uv add numpy
uv add paddleocr
```

**If `uv add paddleocr` fails to resolve/install on this Python (known risk on Python 3.13/Windows):**

```bash
uv add rapidocr-onnxruntime
```

Record in the task report which backend installed. Then verify at least one is importable:

```bash
uv run python -c "
try:
    import paddleocr; print('backend: paddleocr', paddleocr.__version__)
except ImportError:
    import rapidocr_onnxruntime; print('backend: rapidocr')
"
```

Expected: one of the two lines prints. **If NEITHER installs, STOP and report BLOCKED** — do not silently continue without any OCR capability.

- [ ] **Step 2: Write failing tests**

`tests/test_ocr.py`:

```python
from pathlib import Path

import pytest


def test_needs_ocr_thresholds():
    from brain.ocr import needs_ocr

    assert needs_ocr("") is True
    assert needs_ocr("   \n  ") is True
    assert needs_ocr("short") is True          # under 20 chars
    assert needs_ocr("x" * 25) is False        # real text present


def test_ocr_pdf_pages_uses_injected_backend(tmp_path):
    import fitz

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
    import fitz

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
```

Run: `uv run pytest tests/test_ocr.py -v` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

`brain/ocr.py`:

```python
import fitz
import numpy as np

_BACKEND = None


def needs_ocr(page_text: str, min_chars: int = 20) -> bool:
    return len(page_text.strip()) < min_chars


def _load_paddle():
    from paddleocr import PaddleOCR

    try:
        engine = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                           use_doc_unwarping=False, use_textline_orientation=False)
    except TypeError:
        engine = PaddleOCR(lang="en")

    def run(np_img) -> list[str]:
        texts: list[str] = []
        for res in engine.predict(np_img):
            if isinstance(res, dict):
                texts.extend(res.get("rec_texts") or [])
            else:
                texts.extend(getattr(res, "rec_texts", None) or [])
        return texts

    return run


def _load_rapid():
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()

    def run(np_img) -> list[str]:
        result, _elapse = engine(np_img)
        return [item[1] for item in (result or [])]

    return run


def get_ocr_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        _BACKEND = _load_paddle()
    except ImportError:
        _BACKEND = _load_rapid()
    return _BACKEND


def ocr_pdf_pages(path, pages: list[int] | None = None,
                  backend=None, dpi: int = 200) -> dict[int, str]:
    fn = backend or get_ocr_backend()
    doc = fitz.open(path)
    try:
        targets = pages if pages is not None else list(range(len(doc)))
        out: dict[int, str] = {}
        for i in targets:
            pix = doc[i].get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]
            out[i] = "\n".join(fn(img)).strip()
        return out
    finally:
        doc.close()
```

- [ ] **Step 4: Run tests and confirm which backend loaded**

Run: `uv run pytest tests/test_ocr.py -v`
Expected: 4 PASS (or 3 PASS + 1 SKIP if no backend installed — but Step 1 guarantees one exists).
Run: `uv run python -c "from brain.ocr import get_ocr_backend; b = get_ocr_backend(); print('loaded:', b.__qualname__.split('.')[0])"`

- [ ] **Step 5: Commit**

```bash
git add brain/ocr.py tests/test_ocr.py pyproject.toml uv.lock
git commit -m "feat: OCR module - PaddleOCR with RapidOCR fallback, test backend seam"
```

---

### Task 5: Ingestion pipeline (Qdrant + PyMuPDF routing + OCR)

**Files:**
- Create: `brain/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `Settings` (incl. `Settings.index_path`, `Settings.raw_dirs`); `chat_exports.detect_and_parse` (Task 3); `ocr.needs_ocr`, `ocr.ocr_pdf_pages` (Task 4).
- Produces: `build_documents(settings: Settings) -> tuple[list[Document], list[str]]` (documents, warnings); `run_ingest(settings: Settings, rebuild: bool = False, embed_model=None) -> IngestStats`; `IngestStats(files_new: int, files_skipped: int, chunks_indexed: int, warnings: list[str])` with `__str__`; `COLLECTION = "brain"`; CLI via `python -m brain.ingest [--rebuild]`.
- PDF handling contract: every `*.pdf` under `docs/`/`web/` is opened with PyMuPDF; each page becomes one `Document` with metadata `{category, file_name, file_path, page (1-based), file_hash}`; pages where `needs_ocr(page_text)` is True go through `ocr_pdf_pages`; if OCR fails for a file, the file is skipped with a warning (never fatal). Non-PDF files load one-by-one via `SimpleDirectoryReader(input_files=[...])` so one corrupt file warns instead of aborting the batch.

- [ ] **Step 1: Write failing tests**

`tests/test_ingest.py`:

```python
from pathlib import Path

from brain.config import Settings
from brain.ingest import build_documents, run_ingest

NOTE = "# Meeting notes\n\nWe decided the launch date is 2026-11-01.\n\nBudget was approved at 5000 euros."


def make_settings(tmp_path: Path) -> Settings:
    docs = tmp_path / "raw" / "docs"
    chats = tmp_path / "raw" / "chats"
    docs.mkdir(parents=True)
    chats.mkdir(parents=True)
    (docs / "meeting.md").write_text(NOTE, encoding="utf-8")
    (chats / "chat.txt").write_text(
        "15/02/26, 09:14 - Alice: the venue is the old library\n", encoding="utf-8")
    return Settings(raw_dir=tmp_path / "raw", index_path=tmp_path / "index")


class TinyEmbed:  # duck-typed stand-in for a LlamaIndex embed model
    def _embed(self, text: str) -> list[float]:
        import hashlib
        return [b / 255.0 for b in hashlib.md5(text.encode()).digest()[:8]]

    def get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def get_query_embedding(self, text: str) -> list[float]:
        return self._embed(text)


def _make_pdf(path: Path, text: str):
    import fitz

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


def test_rebuild_wipes_index(tmp_path):
    settings = make_settings(tmp_path)
    run_ingest(settings, embed_model=TinyEmbed())
    stats = run_ingest(settings, rebuild=True, embed_model=TinyEmbed())
    assert stats.files_new == 2
```

Run: `uv run pytest tests/test_ingest.py -v` — Expected: FAIL (module missing).

- [ ] **Step 2: Implement**

`brain/ingest.py`:

```python
import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

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
                      "file_path": str(path), "page": i + 1,
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
            try:
                loaded = SimpleDirectoryReader(input_files=[str(f)]).load_data()
                for d in loaded:
                    d.metadata["category"] = category
                    d.metadata["file_hash"] = _file_hash(f)
                    docs.append(d)
            except Exception as e:
                warnings.append(f"{category}: skipped {f.name}: {e}")
    chats_dir = settings.raw_dirs["chats"]
    if chats_dir.exists():
        for f in sorted(chats_dir.rglob("*")):
            if not f.is_file():
                continue
            parsed = chat_exports.detect_and_parse(f)
            if parsed:
                docs.extend(parsed)
            elif f.suffix.lower() in {".txt", ".json"}:
                warnings.append(f"chats: no messages parsed from {f.name}")
    return docs, warnings


def _existing_hashes(client: QdrantClient) -> set[str]:
    hashes: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION, limit=256, offset=offset,
            with_payload=["file_hash"], with_vectors=False)
        for p in points:
            h = (p.payload or {}).get("file_hash")
            if h:
                hashes.add(h)
        if offset is None:
            return hashes


def run_ingest(settings: Settings, rebuild: bool = False, embed_model=None) -> "IngestStats":
    client = QdrantClient(path=str(settings.index_path))
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    all_docs, warnings = build_documents(settings)
    existing = _existing_hashes(client) if client.collection_exists(COLLECTION) else set()
    new_docs = [d for d in all_docs if d.metadata.get("file_hash") not in existing]
    skipped = len(all_docs) - len(new_docs)
    chunks = 0
    if new_docs:
        embed = embed_model or OllamaEmbedding(model_name=settings.embed_model,
                                               base_url=settings.ollama_host)
        splitter = SentenceSplitter(chunk_size=settings.chunk_size,
                                    chunk_overlap=settings.chunk_overlap)
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION)
        storage = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(
            new_docs, storage_context=storage, embed_model=embed,
            transformations=[splitter], show_progress=False)
        chunks = client.count(COLLECTION, exact=True).count
    client.close()
    return IngestStats(files_new=len(new_docs), files_skipped=skipped,
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
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 6 PASS. (If `TinyEmbed` duck-typing is rejected by LlamaIndex internals, replace it with a subclass of `llama_index.core.base.embeddings.base.BaseEmbedding` implementing `_get_query_embedding`, `_get_text_embedding`, `_aget_query_embedding` with the same md5 logic.)

- [ ] **Step 4: Commit**

```bash
git add brain/ingest.py tests/test_ingest.py
git commit -m "feat: incremental ingestion into Qdrant with PyMuPDF routing and OCR fallback"
```

---

### Task 6: Hybrid query engine (dense + BM25, RRF fusion)

**Files:**
- Create: `brain/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `Settings`, `brain.persona.build_system_prompt`, `brain.persona.REFUSAL`, `OllamaClient.chat` / `.chat_stream` (Task 2), `COLLECTION` convention from Task 5.
- Produces:
  - `Source(file: str, snippet: str, score: float)`, `QueryResult(answer: str, sources: list[Source], timing: dict)`
  - `EmptyIndexError(RuntimeError)`
  - `BrainEngine(index, client, settings, corpus=None)` where `corpus: dict[str, tuple[str, dict]]` maps Qdrant point id → `(text, metadata)`; `None` → dense-only (unit tests). Methods: `query(question: str, top_k: int | None = None, deep_think: bool = False) -> QueryResult`, `query_stream(...) -> Iterator[dict]` (yields `{"type": "delta", "text": ...}` then a final `{"type": "final", "sources": [...], "timing": {...}}`), `chunk_count() -> int`
  - `load_engine(settings: Settings, embed_model=None, client=None) -> BrainEngine` — opens Qdrant at `settings.index_path`, raises `EmptyIndexError` when the collection is missing/empty, scrolls the corpus and builds the BM25 index.
- Retrieval contract: dense retriever fetches `2×top_k`; BM25 searches the corpus for `2×top_k`; fusion score = `Σ 1/(60 + rank + 1)` over each channel the chunk appears in; top `top_k` fused chunks go into the prompt. A chunk counts as "strong evidence" if `dense_score >= min_score` OR it has a BM25 hit (the exact terms are literally present). **Evidence gate (user flowchart): when NO fused chunk is strong, generation is skipped entirely — `query` returns `QueryResult(answer=persona.REFUSAL, sources=...)` and `query_stream` yields the refusal as the only delta — with the nearest sources + scores still attached so the caller can judge.** `Source.score` reports the dense similarity (0.0 for BM25-only hits).

- [ ] **Step 1: Write failing tests**

`tests/test_engine.py`:

```python
from pathlib import Path

from brain.config import Settings
from brain.engine import BrainEngine, EmptyIndexError, QueryResult, Source
from brain.ollama_client import ChatResult


class FakeNode:
    def __init__(self, id_, text, score, meta):
        self.id_, self.text, self.score, self.metadata = id_, text, score, meta

    def get_content(self):
        return self.text


class FakeRetriever:
    def __init__(self, nodes):
        self.nodes = nodes

    def retrieve(self, q):
        return self.nodes


class FakeIndex:
    def __init__(self, nodes):
        self.nodes = nodes

    def as_retriever(self, similarity_top_k=5):
        return FakeRetriever(self.nodes)


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        return ChatResult(text="The launch date is 2026-11-01 [meeting.md].",
                          thinking="", eval_count=10, tok_per_s=15.0)

    def chat_stream(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        yield from ["The launch ", "date is 2026-11-01."]


def make_engine(settings, client=None, corpus=None):
    nodes = [FakeNode("p1", "We decided the launch date is 2026-11-01.", 0.82,
                      {"file_name": "meeting.md"}),
             FakeNode("p2", "Venue is the old library.", 0.61, {"file_name": "chat.txt"})]
    return BrainEngine(FakeIndex(nodes), client or FakeClient(), settings, corpus=corpus)


def make_settings(tmp_path):
    return Settings(raw_dir=tmp_path, index_path=tmp_path / "idx")


def test_query_returns_answer_sources_timing(tmp_path):
    eng = make_engine(make_settings(tmp_path))
    res = eng.query("when is the launch?")
    assert isinstance(res, QueryResult)
    assert "2026-11-01" in res.answer
    assert [src.file for src in res.sources] == ["meeting.md", "chat.txt"]
    assert res.sources[0].score == 0.82
    assert set(res.timing) == {"retrieve_ms", "generate_ms", "tokens", "tok_per_s"}


def test_query_default_uses_no_thinking(tmp_path):
    s = make_settings(tmp_path)
    client = FakeClient()
    eng = make_engine(s, client)
    eng.query("q")
    call = client.calls[0]
    assert call["think"] is False
    assert call["max_tokens"] is None
    assert call["keep_alive"] == s.keep_alive


def test_deep_think_caps_tokens(tmp_path):
    s = make_settings(tmp_path)
    client = FakeClient()
    eng = make_engine(s, client)
    eng.query("hard question", deep_think=True)
    call = client.calls[0]
    assert call["think"] is True
    assert call["max_tokens"] == s.deep_think_max_tokens


def test_system_prompt_contains_context_and_rules(tmp_path):
    s = make_settings(tmp_path)
    client = FakeClient()
    eng = make_engine(s, client)
    eng.query("q")
    sys_msg = client.calls[0]["messages"][0]
    assert sys_msg["role"] == "system"
    assert "meeting.md" in sys_msg["content"]      # context included
    assert "ONLY" in sys_msg["content"]            # grounding rules included


def test_insufficient_evidence_short_circuits_with_refusal(tmp_path):
    # No chunk above min_score and no BM25 hit -> canned refusal, NO generation call,
    # but nearest sources still returned so the caller can judge.
    s = Settings(raw_dir=tmp_path, index_path=tmp_path / "idx", min_score=0.9)
    client = FakeClient()
    nodes = [FakeNode("p1", "irrelevant", 0.10, {"file_name": "a.md"})]
    eng = BrainEngine(FakeIndex(nodes), client, s)
    res = eng.query("q")
    assert res.answer == "I don't know - that's not in your documents."
    assert res.sources[0].file == "a.md"
    assert client.calls == []
    assert res.timing["tokens"] == 0


def test_bm25_surfaces_exact_match_missing_from_dense(tmp_path):
    s = make_settings(tmp_path)
    nodes = [FakeNode("p1", "We decided the launch date is 2026-11-01.", 0.40,
                      {"file_name": "meeting.md"})]
    corpus = {"p2": ("Invoice 4471 total is 88 euros", {"file_name": "invoice.txt"})}
    eng = BrainEngine(FakeIndex(nodes), FakeClient(), s, corpus=corpus)
    res = eng.query("4471")
    files = [src.file for src in res.sources]
    assert "invoice.txt" in files


def test_rrf_ranks_dual_channel_hits_first(tmp_path):
    s = make_settings(tmp_path)
    nodes = [FakeNode("p1", "We decided the launch date is 2026-11-01.", 0.80,
                      {"file_name": "meeting.md"}),
             FakeNode("p2", "Venue is the old library.", 0.50, {"file_name": "dual.txt"})]
    corpus = {"p2": ("Venue is the old library.", {"file_name": "dual.txt"})}
    eng = BrainEngine(FakeIndex(nodes), FakeClient(), s, corpus=corpus)
    res = eng.query("venue old library")
    assert res.sources[0].file == "dual.txt"  # in both channels beats dense-only p1


def test_bm25_hit_counts_as_strong_evidence(tmp_path):
    # min_score is huge, but the BM25 channel matched -> generation proceeds anyway
    s = Settings(raw_dir=tmp_path, index_path=tmp_path / "idx", min_score=0.99)
    nodes = [FakeNode("p1", "unrelated filler text here", 0.10, {"file_name": "a.md"})]
    corpus = {"p2": ("Invoice 4471 total is 88 euros", {"file_name": "invoice.txt"})}
    client = FakeClient()
    eng = BrainEngine(FakeIndex(nodes), client, s, corpus=corpus)
    res = eng.query("4471")
    assert res.answer != "I don't know - that's not in your documents."
    sys_msg = client.calls[0]["messages"][0]["content"]
    assert "invoice.txt" in sys_msg


def test_query_stream(tmp_path):
    eng = make_engine(make_settings(tmp_path))
    events = list(eng.query_stream("when is the launch?"))
    assert events[0]["type"] == "delta"
    assert events[-1]["type"] == "final"
    assert events[-1]["sources"][0].file == "meeting.md"


def test_query_stream_short_circuits_without_generation(tmp_path):
    s = Settings(raw_dir=tmp_path, index_path=tmp_path / "idx", min_score=0.9)
    client = FakeClient()
    nodes = [FakeNode("p1", "irrelevant", 0.10, {"file_name": "a.md"})]
    eng = BrainEngine(FakeIndex(nodes), client, s)
    events = list(eng.query_stream("q"))
    assert events[0] == {"type": "delta", "text": "I don't know - that's not in your documents."}
    assert events[-1]["type"] == "final"
    assert events[-1]["sources"][0]["file"] == "a.md"
    assert client.calls == []


def test_load_engine_raises_on_missing_index(tmp_path):
    from brain.engine import load_engine

    s = Settings(raw_dir=tmp_path, index_path=tmp_path / "does-not-exist")
    try:
        load_engine(s)
        raise AssertionError("expected EmptyIndexError")
    except EmptyIndexError:
        pass
```

Run: `uv run pytest tests/test_engine.py -v` — Expected: FAIL (module missing).

- [ ] **Step 2: Implement**

`brain/engine.py`:

```python
import json
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from rank_bm25 import BM25Okapi

from brain import persona
from brain.config import Settings
from brain.ollama_client import OllamaClient

COLLECTION = "brain"
_RRF_K = 60


class EmptyIndexError(RuntimeError):
    pass


@dataclass
class Source:
    file: str
    snippet: str
    score: float


@dataclass
class QueryResult:
    answer: str
    sources: list[Source]
    timing: dict = field(default_factory=dict)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _source_file(meta: dict) -> str:
    base = meta.get("file_name") or meta.get("chat_name") or meta.get("file_path") or "unknown"
    page = meta.get("page")
    return f"{base}#p{page}" if page else base


class _BM25:
    def __init__(self, corpus: dict[str, tuple[str, dict]]):
        self._ids = list(corpus)
        docs = [_tokenize(corpus[pid][0]) for pid in self._ids]
        self._bm25 = BM25Okapi(docs) if docs and any(docs) else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._ids[i], scores[i]) for i in ranked if scores[i] > 0]


class BrainEngine:
    def __init__(self, index, client: OllamaClient, settings: Settings,
                 corpus: dict[str, tuple[str, dict]] | None = None):
        self._index = index
        self._client = client
        self._settings = settings
        self._corpus = corpus or {}
        self._bm25 = _BM25(self._corpus) if self._corpus else None

    @classmethod
    def load(cls, settings: Settings, embed_model=None,
             client: OllamaClient | None = None) -> "BrainEngine":
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        qdrant = QdrantClient(path=str(settings.index_path))
        if not qdrant.collection_exists(COLLECTION) or \
                qdrant.count(COLLECTION, exact=True).count == 0:
            qdrant.close()
            raise EmptyIndexError(
                f"No index at {settings.index_path}. Run: uv run python -m brain.ingest")
        embed = embed_model or OllamaEmbedding(model_name=settings.embed_model,
                                               base_url=settings.ollama_host)
        store = QdrantVectorStore(client=qdrant, collection_name=COLLECTION)
        index = VectorStoreIndex.from_vector_store(store, embed_model=embed)
        corpus = _scroll_corpus(qdrant, COLLECTION)
        ollama = client or OllamaClient(settings.ollama_host, settings.llm_model)
        return cls(index, ollama, settings, corpus=corpus)

    def _retrieve(self, question: str, top_k: int):
        dense = self._index.as_retriever(similarity_top_k=top_k * 2).retrieve(question)
        cand: dict[str, dict] = {}
        for rank, n in enumerate(dense):
            pid = str(n.id_)
            cand[pid] = {"text": n.get_content(), "metadata": dict(n.metadata),
                         "dense_rank": rank, "dense_score": float(n.score or 0.0),
                         "bm25_rank": None}
        if self._bm25 is not None:
            for rank, (pid, _s) in enumerate(self._bm25.search(question, top_k * 2)):
                if pid in cand:
                    cand[pid]["bm25_rank"] = rank
                elif pid in self._corpus:
                    text, meta = self._corpus[pid]
                    cand[pid] = {"text": text, "metadata": meta, "dense_rank": None,
                                 "dense_score": 0.0, "bm25_rank": rank}

        def rrf(c: dict) -> float:
            score = 0.0
            if c["dense_rank"] is not None:
                score += 1.0 / (_RRF_K + c["dense_rank"] + 1)
            if c["bm25_rank"] is not None:
                score += 1.0 / (_RRF_K + c["bm25_rank"] + 1)
            return score

        fused = sorted(cand.items(), key=lambda kv: rrf(kv[1]), reverse=True)[:top_k]

        def strong(c: dict) -> bool:
            return c["dense_score"] >= self._settings.min_score or c["bm25_rank"] is not None

        sources = [Source(file=_source_file(c["metadata"]), snippet=c["text"][:200],
                          score=round(c["dense_score"], 3)) for _, c in fused]
        context = "\n\n---\n\n".join(
            f"[{_source_file(c['metadata'])}]\n{c['text']}" for _, c in fused)
        has_strong = any(strong(c) for _, c in fused) if fused else False
        return context, sources, has_strong

    def query(self, question: str, top_k: int | None = None,
              deep_think: bool = False) -> QueryResult:
        k = top_k or self._settings.top_k
        t0 = time.perf_counter()
        context, sources, has_strong = self._retrieve(question, k)
        t1 = time.perf_counter()
        if not has_strong:
            return QueryResult(answer=persona.REFUSAL, sources=sources,
                               timing={"retrieve_ms": round((t1 - t0) * 1000),
                                       "generate_ms": 0, "tokens": 0, "tok_per_s": 0.0})
        messages = [
            {"role": "system", "content": persona.build_system_prompt(context)},
            {"role": "user", "content": question},
        ]
        res = self._client.chat(
            messages, think=deep_think,
            max_tokens=self._settings.deep_think_max_tokens if deep_think else None,
            keep_alive=self._settings.keep_alive)
        t2 = time.perf_counter()
        timing = {"retrieve_ms": round((t1 - t0) * 1000), "generate_ms": round((t2 - t1) * 1000),
                  "tokens": res.eval_count, "tok_per_s": res.tok_per_s}
        return QueryResult(answer=res.text, sources=sources, timing=timing)

    def query_stream(self, question: str, top_k: int | None = None,
                     deep_think: bool = False) -> Iterator[dict]:
        k = top_k or self._settings.top_k
        t0 = time.perf_counter()
        context, sources, has_strong = self._retrieve(question, k)
        t1 = time.perf_counter()
        first = {"retrieve_ms": round((t1 - t0) * 1000)}
        if not has_strong:
            yield {"type": "delta", "text": persona.REFUSAL}
            yield {"type": "final", "sources": [asdict(s) for s in sources], "timing": first}
            return
        messages = [
        for delta in self._client.chat_stream(
                messages, think=deep_think,
                max_tokens=self._settings.deep_think_max_tokens if deep_think else None,
                keep_alive=self._settings.keep_alive):
            yield {"type": "delta", "text": delta}
        yield {"type": "final", "sources": [asdict(s) for s in sources], "timing": first}

    def chunk_count(self) -> int:
        try:
            from qdrant_client import QdrantClient

            c = QdrantClient(path=str(self._settings.index_path))
            n = c.count(COLLECTION, exact=True).count
            c.close()
            return n
        except Exception:
            return 0


def _scroll_corpus(qdrant, collection: str) -> dict[str, tuple[str, dict]]:
    corpus: dict[str, tuple[str, dict]] = {}
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection, limit=256, offset=offset,
            with_payload=True, with_vectors=False)
        for p in points:
            raw = (p.payload or {}).get("_node_content")
            if not raw:
                continue
            node = json.loads(raw)
            corpus[str(p.id)] = (node.get("text", ""), node.get("metadata") or {})
        if offset is None:
            return corpus


def load_engine(settings: Settings, embed_model=None,
                client: OllamaClient | None = None) -> BrainEngine:
    return BrainEngine.load(settings, embed_model=embed_model, client=client)
```

- [ ] **Step 3: Run all tests so far**

Run: `uv run pytest -v`
Expected: all previous tasks' tests still PASS, plus 11 new engine tests PASS.

- [ ] **Step 4: Commit**

```bash
git add brain/engine.py tests/test_engine.py
git commit -m "feat: hybrid retrieval engine - dense + BM25 fused via RRF, grounded streaming"
```

---

### Task 7: FastAPI service + ask script

**Files:**
- Create: `brain/api.py`, `scripts/ask.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `BrainEngine`, `QueryResult`, `EmptyIndexError`, `OllamaDown`, `get_settings`.
- Produces: `create_app(engine=None, settings=None) -> FastAPI` with `GET /health`, `POST /query`, `POST /query/stream`; `python -m brain.api` starts uvicorn on `settings.api_port`; `scripts/ask.py "question" [--deep]` prints answer + sources.

- [ ] **Step 1: Write failing tests**

`tests/test_api.py`:

```python
import json

from fastapi.testclient import TestClient

from brain.api import create_app
from brain.engine import EmptyIndexError, QueryResult, Source
from brain.ollama_client import OllamaDown


class FakeEngine:
    def __init__(self, fail=None):
        self.fail = fail
        self.last_query = None

    def query(self, question, top_k=None, deep_think=False):
        self.last_query = {"question": question, "top_k": top_k, "deep_think": deep_think}
        if self.fail == "down":
            raise OllamaDown("unreachable")
        return QueryResult(
            answer="The venue is the old library [chat.txt].",
            sources=[Source(file="chat.txt", snippet="the venue is...", score=0.8)],
            timing={"retrieve_ms": 5, "generate_ms": 900, "tokens": 40, "tok_per_s": 44.0})

    def query_stream(self, question, top_k=None, deep_think=False):
        yield {"type": "delta", "text": "The venue "}
        yield {"type": "delta", "text": "is the old library."}
        yield {"type": "final", "sources": [{"file": "chat.txt", "snippet": "s", "score": 0.8}],
               "timing": {"retrieve_ms": 5}}

    def chunk_count(self):
        return 42


def test_health():
    client = TestClient(create_app(engine=FakeEngine()))
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] == 42


def test_query_ok_and_forwards_flags():
    eng = FakeEngine()
    client = TestClient(create_app(engine=eng))
    r = client.post("/query", json={"question": "where is the venue?", "top_k": 3, "deep_think": True})
    assert r.status_code == 200
    body = r.json()
    assert "old library" in body["answer"]
    assert body["sources"][0]["file"] == "chat.txt"
    assert eng.last_query == {"question": "where is the venue?", "top_k": 3, "deep_think": True}


def test_query_validation_error():
    client = TestClient(create_app(engine=FakeEngine()))
    assert client.post("/query", json={}).status_code == 422


def test_query_ollama_down_is_503():
    client = TestClient(create_app(engine=FakeEngine(fail="down")))
    r = client.post("/query", json={"question": "q"})
    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]


def test_query_empty_index_is_409():
    class EmptyEng(FakeEngine):
        def query(self, *a, **kw):
            raise EmptyIndexError("no index")
    client = TestClient(create_app(engine=EmptyEng()))
    assert client.post("/query", json={"question": "q"}).status_code == 409


def test_stream_emits_sse_events():
    client = TestClient(create_app(engine=FakeEngine()))
    r = client.post("/query/stream", json={"question": "where is the venue?"})
    assert r.status_code == 200
    text = r.text
    deltas = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ") and "delta" in l]
    final = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ") and "final" in l]
    assert "".join(d["text"] for d in deltas) == "The venue is the old library."
    assert final and final[0]["sources"][0]["file"] == "chat.txt"
```

Run: `uv run pytest tests/test_api.py -v` — Expected: FAIL (module missing).

- [ ] **Step 2: Implement api.py**

`brain/api.py`:

```python
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from brain.config import get_settings
from brain.engine import EmptyIndexError
from brain.ollama_client import OllamaDown


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None
    deep_think: bool = False


def create_app(engine=None, settings=None) -> FastAPI:
    settings = settings or get_settings()
    if engine is None:
        from brain.engine import load_engine
        engine = load_engine(settings)
    app = FastAPI(title="brain")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": settings.llm_model,
                "embed_model": settings.embed_model, "indexed_chunks": engine.chunk_count()}

    @app.post("/query")
    def query(req: QueryRequest):
        try:
            res = engine.query(req.question, top_k=req.top_k, deep_think=req.deep_think)
        except OllamaDown as e:
            raise HTTPException(503, f"Ollama is not reachable: {e}. Start the Ollama app.")
        except EmptyIndexError as e:
            raise HTTPException(409, str(e))
        return {"answer": res.answer,
                "sources": [{"file": s.file, "snippet": s.snippet, "score": s.score} for s in res.sources],
                "timing": res.timing}

    @app.post("/query/stream")
    def query_stream(req: QueryRequest):
        def sse():
            try:
                for event in engine.query_stream(req.question, top_k=req.top_k,
                                                 deep_think=req.deep_think):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except OllamaDown as e:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


def main() -> None:
    import uvicorn
    settings = get_settings()
    uvicorn.run(create_app(settings=settings), host="127.0.0.1", port=settings.api_port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Implement scripts/ask.py**

`scripts/ask.py`:

```python
import argparse

from brain.config import get_settings
from brain.engine import load_engine


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask your brain a question")
    ap.add_argument("question")
    ap.add_argument("--deep", action="store_true", help="enable thinking mode (slower)")
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()
    engine = load_engine(get_settings())
    res = engine.query(args.question, top_k=args.top_k, deep_think=args.deep)
    print(res.answer)
    if res.sources:
        print("\nSources:")
        for s in res.sources:
            print(f"  [{s.score:.2f}] {s.file}: {s.snippet[:100]}...")
    t = res.timing
    print(f"\n({t['retrieve_ms']}ms retrieve, {t['generate_ms']}ms generate, "
          f"{t['tokens']} tokens @ {t['tok_per_s']} tok/s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run full offline suite**

Run: `uv run pytest -v`
Expected: ALL PASS (config, client, parsers, ocr, ingest, engine, api).

- [ ] **Step 5: Commit**

```bash
git add brain/api.py scripts/ask.py tests/test_api.py
git commit -m "feat: FastAPI service with health/query/stream and ask script"
```

---

### Task 8: Live end-to-end smoke test

**Files:**
- Create: `tests/test_live.py`, `data/raw/docs/getting_started.md` (sample note, user-replaceable)

**Interfaces:**
- Consumes: everything above, against the real Ollama server and models.
- Produces: confidence that the full loop works on this machine, plus measured latency numbers for the final report.

- [ ] **Step 1: Pull the embedding model**

```bash
ollama pull qwen3-embedding:0.6b
ollama list
```

Expected: `qwen3-embedding:0.6b` listed (~639 MB).

- [ ] **Step 2: Create a sample note and ingest**

`data/raw/docs/getting_started.md`:

```markdown
# Brain getting-started notes

- This project is the personal AI brain. Phase 1 is RAG over my own files.
- The engine model is qwen3.5:4b running locally through Ollama.
- My machine has an RTX 3050 with 4GB VRAM and 16GB RAM.
```

```bash
uv run python -m brain.ingest
```

Expected output like `new files: 1, skipped: 0, chunks indexed: 1`. (API server must NOT be running — Qdrant local mode allows one process.)

- [ ] **Step 3: Write the live smoke test**

`tests/test_live.py`:

```python
import pytest

from brain.config import get_settings
from brain.engine import load_engine
from brain.ollama_client import OllamaClient

pytestmark = pytest.mark.skipif(
    not OllamaClient(get_settings().ollama_host, get_settings().llm_model).is_up(),
    reason="Ollama not running")


@pytest.fixture(scope="module")
def engine():
    return load_engine(get_settings())


def test_in_domain_answer_cites_source(engine):
    res = engine.query("What model powers the personal AI brain?")
    assert "qwen3.5:4b" in res.answer
    assert any("getting_started" in s.file for s in res.sources)


def test_out_of_domain_refuses(engine):
    res = engine.query("What is the capital of France?")
    assert "not in your documents" in res.answer.lower()


def test_exact_number_query_finds_source(engine):
    res = engine.query("How much VRAM does my machine have?")
    assert res.answer  # non-empty
    assert res.sources  # at least one chunk retrieved


def test_deep_think_runs(engine):
    res = engine.query("What model powers the personal AI brain?", deep_think=True)
    assert res.answer  # non-empty; capped at deep_think_max_tokens
```

- [ ] **Step 4: Run the live suite and the ask script**

```bash
uv run pytest tests/test_live.py -v
uv run python scripts/ask.py "What hardware does the brain run on?"
uv run python scripts/ask.py "What is the capital of France?"
```

Expected: live tests PASS (in-domain cited, out-of-domain refused). `ask.py` answers the first from the note with a citation and timing line; refuses the second. Record the timing numbers for the final report. **If the out-of-domain case answers anyway, the grounding prompt needs strengthening, not a silent pass — report it.**

- [ ] **Step 5: Exercise the API manually**

```bash
uv run python -m brain.api   # in a separate terminal if interactive
curl -s http://localhost:8400/health
curl -s -X POST http://localhost:8400/query -H "Content-Type: application/json" -d "{\"question\": \"What model powers the personal AI brain?\"}"
```

Expected: health JSON shows `indexed_chunks >= 1`; query returns the cited answer. Stop the API server afterward (single-process Qdrant).

- [ ] **Step 6: Final commit**

```bash
git add tests/test_live.py data/raw/docs/getting_started.md
git commit -m "test: live end-to-end smoke suite against real Ollama"
```

---

## Non-goals (per spec §10)

Web UI, phone bot, multi-user/auth, cloud, agent tool-calling, folder watching, reranker (parked by user), model A/B harness (optional follow-up), Phase 2 fine-tuning (separate spec/plan).

## Open items this plan deliberately does not resolve (spec §11)

Third chat app parser and real document locations — neither blocks Phase 1 code; both block *real-data* ingestion value. Language mix is resolved (multilingual `qwen3-embedding:0.6b`).
