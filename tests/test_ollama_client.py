import json

import httpx
import pytest

from brain.ollama_client import OllamaClient, OllamaDown, OllamaError


def make_client(handler) -> OllamaClient:
    return OllamaClient("http://fake", "phi4-mini", transport=httpx.MockTransport(handler))


def test_chat_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "phi4-mini"
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


def test_chat_malformed_json_raises_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    c = make_client(handler)
    with pytest.raises(OllamaError, match="malformed"):
        c.chat([{"role": "user", "content": "q"}])


def test_stream_malformed_line_raises_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{broken line\n")

    c = make_client(handler)
    with pytest.raises(OllamaError, match="malformed"):
        list(c.chat_stream([{"role": "user", "content": "q"}]))


def test_chat_empty_answer_with_think_raises_loud_error():
    # A model's thinking loop can exhaust the whole num_predict budget and
    # return content="" - that must be a LOUD failure, never a silent "".
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {"content": ""}, "thinking": "loop loop loop",
            "eval_count": 3072, "eval_duration": 240_000_000_000})

    c = make_client(handler)
    with pytest.raises(OllamaError, match="BRAIN_DEEP_THINK_MAX_TOKENS"):
        c.chat([{"role": "user", "content": "q"}], think=True, max_tokens=3072)


def test_chat_empty_answer_without_think_is_returned_as_is():
    # The guard is scoped to think mode (the proven failure path).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {"content": ""}, "eval_count": 5, "eval_duration": 1})

    c = make_client(handler)
    res = c.chat([{"role": "user", "content": "q"}], think=False)
    assert res.text == ""
