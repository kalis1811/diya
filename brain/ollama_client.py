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
        try:
            d = r.json()
        except ValueError as e:
            raise OllamaError(f"malformed JSON from Ollama (chat): {e}") from e
        msg = d.get("message", {})
        count = d.get("eval_count", 0)
        dur = d.get("eval_duration", 0)
        text = msg.get("content", "")
        if think and not text.strip():
            budget = max_tokens if max_tokens is not None else "default"
            raise OllamaError(
                "model returned no answer - thinking consumed the entire "
                f"{budget}-token budget; raise BRAIN_DEEP_THINK_MAX_TOKENS or retry")
        return ChatResult(
            text=text,
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
                        try:
                            d = json.loads(line)
                        except ValueError as e:
                            raise OllamaError(
                                f"malformed line from Ollama stream: {line[:80]!r}") from e
                        delta = d.get("message", {}).get("content", "")
                        if delta:
                            yield delta
                        if d.get("done"):
                            break
        except httpx.HTTPError as e:
            raise OllamaDown(f"Ollama unreachable at {self._base}: {e}") from e
