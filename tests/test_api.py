import base64
import dataclasses
import json

from fastapi.testclient import TestClient

from brain.api import create_app
from brain.config import Settings, get_settings
from brain.engine import EmptyIndexError, QueryResult, Source
from brain.ollama_client import OllamaDown, OllamaError

from conftest import TEST_ADMIN_PASSWORD


def make_client(**kwargs) -> TestClient:
    """Authenticated TestClient: the admin middleware guards every non-/health
    route, so tests must present valid Basic credentials for the same settings
    object the app was built with."""
    settings = kwargs.pop("settings", None) or get_settings()
    if len(settings.admin_password) < 10:
        settings = dataclasses.replace(settings, admin_password=TEST_ADMIN_PASSWORD)
    client = TestClient(create_app(settings=settings, **kwargs))
    token = base64.b64encode(
        f"{settings.admin_username}:{settings.admin_password}".encode("utf-8")).decode("ascii")
    client.headers["Authorization"] = f"Basic {token}"
    return client


class FakeEngine:
    def __init__(self, fail=None):
        self.fail = fail
        self.last_query = None

    def query(self, question, top_k=None, deep_think=False):
        self.last_query = {"question": question, "top_k": top_k, "deep_think": deep_think}
        if self.fail == "down":
            raise OllamaDown("unreachable")
        if self.fail == "model_error":
            raise OllamaError("model returned an error")
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
    client = make_client(engine=FakeEngine())
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] == 42


def test_app_starts_with_empty_index_and_reports_ingest_hint(tmp_path):
    settings = Settings(index_path=tmp_path / "missing-index", raw_dir=tmp_path / "raw")
    client = make_client(settings=settings)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "needs_ingest"
    assert health.json()["indexed_chunks"] == 0

    query = client.post("/query", json={"question": "What did I tell you?"})
    assert query.status_code == 200
    assert query.json()["answer"] == "I don't know - that's not in your documents."

    stream = client.post("/query/stream", json={"question": "What did I tell you?"})
    assert stream.status_code == 200
    assert "I don't know - that's not in your documents." in stream.text


def test_browser_action_opens_without_a_knowledge_index(tmp_path):
    opened = []
    settings = Settings(index_path=tmp_path / "missing-index", raw_dir=tmp_path / "raw")
    client = make_client(
        settings=settings,
        browser_opener=lambda url, **kw: opened.append(url) or True,
    )
    response = client.post("/query", json={"question": "open YouTube"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Opened YouTube in your default browser."
    assert opened == ["https://www.youtube.com/"]

    events = client.post("/query/stream", json={"question": "open Google"})
    assert events.status_code == 200
    assert "Opened Google in your default browser." in events.text
    assert opened[-1] == "https://www.google.com/"


def test_query_ok_and_forwards_flags():
    eng = FakeEngine()
    client = make_client(engine=eng)
    r = client.post("/query", json={"question": "where is the venue?", "top_k": 3, "deep_think": True})
    assert r.status_code == 200
    body = r.json()
    assert "old library" in body["answer"]
    assert body["sources"][0]["file"] == "chat.txt"
    assert eng.last_query == {"question": "where is the venue?", "top_k": 3, "deep_think": True}


def test_query_validation_error():
    client = make_client(engine=FakeEngine())
    assert client.post("/query", json={}).status_code == 422
    assert client.post("/query", json={"question": "   "}).status_code == 422
    assert client.post("/query", json={"question": "q", "top_k": 0}).status_code == 422
    assert client.post("/query", json={"question": "q", "top_k": 21}).status_code == 422


def test_query_ollama_down_is_503():
    client = make_client(engine=FakeEngine(fail="down"))
    r = client.post("/query", json={"question": "q"})
    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]


def test_query_ollama_error_is_503():
    client = make_client(engine=FakeEngine(fail="model_error"))
    r = client.post("/query", json={"question": "q"})
    assert r.status_code == 503
    assert "model returned an error" in r.json()["detail"]


def test_query_empty_index_is_409():
    class EmptyEng(FakeEngine):
        def query(self, *a, **kw):
            raise EmptyIndexError("no index")
    client = make_client(engine=EmptyEng())
    assert client.post("/query", json={"question": "q"}).status_code == 409


def test_stream_emits_sse_events():
    client = make_client(engine=FakeEngine())
    r = client.post("/query/stream", json={"question": "where is the venue?"})
    assert r.status_code == 200
    text = r.text
    deltas = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ") and "delta" in l]
    final = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ") and "final" in l]
    assert "".join(d["text"] for d in deltas) == "The venue is the old library."
    assert final and final[0]["sources"][0]["file"] == "chat.txt"


def test_stream_ollama_error_emits_error_event():
    class BrokenEng(FakeEngine):
        def query_stream(self, question, top_k=None, deep_think=False):
            yield {"type": "delta", "text": "partial"}
            raise OllamaError("model x returned 500")

    client = make_client(engine=BrokenEng())
    r = client.post("/query/stream", json={"question": "q"})
    assert r.status_code == 200
    assert "error" in r.text and "model x returned 500" in r.text
