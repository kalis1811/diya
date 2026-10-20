import json
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from rank_bm25 import BM25Okapi

from brain import persona
from brain.browser_actions import run_browser_command
from brain.config import Settings
from brain.memory import parse_remember_command
from brain.storage import get_storage
from brain.ollama_client import OllamaClient, OllamaDown
from brain.scheduling import appointment_schedule

COLLECTION = "brain"
_RRF_K = 60

# Stopwords are stripped from BM25 tokens (index and query sides): stopword
# overlap would otherwise open the evidence gate for any English sentence pair.
_STOPWORDS = frozenset(
    "a an and are as at be been but by did do does for from has have he her his "
    "how i in is it its my of on or she that the their them they this to was we "
    "were what when where which who why will with you your".split())
_QUERY_WRAPPERS = frozenset(
    "according chat chats file files note notes document documents pdf image images "
    "message messages say says said tell told me please based about explain summarize "
    "summarize describe using from in my own".split())
_FOLLOW_UP_RE = re.compile(
    r"^(?:and\b|what about\b|how about\b|why\b|how so\b|tell me more\b|"
    r"what do you mean\b|can you explain\b|it\b|that\b|they\b|those\b|these\b|this\b)",
    re.IGNORECASE)
_SOURCE_REQUEST_RE = re.compile(
    r"\b(?:according\s+to|from|in)\s+(?:my\s+)?(?:files?|documents?|notes?|chats?|history)\b",
    re.IGNORECASE)
_SELF_CONTAINED_SCHEDULE_RE = re.compile(
    r"(?=.*\b(?:schedule|plan)\b)(?=.*\bappointment\b)"
    r"(?=.*\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b)"
    r"(?=.*\b(?:travel|drive|trip|commute)\b.{0,50}\b\d+\s*"
    r"(?:minutes?|mins?|hours?|hrs?)\b)",
    re.IGNORECASE | re.DOTALL)


def _retrieval_question(question: str, history: list[dict] | None) -> str:
    """Add the last user question to short follow-ups so retrieval keeps its topic."""
    if not history:
        return question
    prior = next((item.get("content", "") for item in reversed(history)
                  if item.get("role") == "user" and item.get("content")), "")
    short_or_referential = len(question.split()) <= 5 or _FOLLOW_UP_RE.search(question.strip())
    return f"{prior}\n{question}" if prior and short_or_referential else question


def _needs_personal_retrieval(question: str) -> bool:
    # When the user supplies the appointment and travel times and asks for a
    # schedule, solve from those facts instead of retrieving unrelated chat
    # templates or instructions. Explicit questions about stored sources still
    # use retrieval.
    if (_SELF_CONTAINED_SCHEDULE_RE.search(question) and
            not _SOURCE_REQUEST_RE.search(question)):
        return False
    return persona.is_personal_fact_question(question)


def should_use_deep_thinking(question: str, requested: bool = False,
                             model: str = "") -> bool:
    """Enable Ollama thinking only for models known to support the option."""
    if not requested:
        return False
    normalized = model.casefold().split(":", 1)[0]
    return normalized.startswith(("qwen3", "deepseek-r1"))


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
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


def _content_terms(question: str) -> set[str]:
    """Remove phrasing used to request personal context, keeping topic terms."""
    return set(_tokenize(question)) - _QUERY_WRAPPERS


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
        # score is exactly 0.0 iff the doc contains none of the query terms;
        # tiny corpora can give matched terms negative IDF, so >0 would drop real hits
        return [(self._ids[i], scores[i]) for i in ranked if scores[i] != 0]


class BrainEngine:
    def __init__(self, index, client: OllamaClient, settings: Settings,
                 corpus: dict[str, tuple[str, dict]] | None = None,
                 qdrant=None, browser_opener=None):
        self._index = index
        self._client = client
        self._settings = settings
        self._corpus = corpus or {}
        self._bm25 = _BM25(self._corpus) if self._corpus else None
        self._qdrant = qdrant
        self._browser_opener = browser_opener

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
        return cls(index, ollama, settings, corpus=corpus, qdrant=qdrant)

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

        intent_terms = _content_terms(question)
        if intent_terms:
            # Dense chat embeddings often rank unrelated messages highly because
            # they share tone or broad context. Require up to two topic terms so
            # a coincidental word does not expose an unrelated private message.
            min_term_overlap = min(2, len(intent_terms))
            cand = {
                pid: candidate for pid, candidate in cand.items()
                if len(intent_terms & set(_tokenize(candidate["text"]))) >= min_term_overlap
            }
        fused = sorted(cand.items(), key=lambda kv: rrf(kv[1]), reverse=True)[:top_k]

        def strong(c: dict) -> bool:
            return c["dense_score"] >= self._settings.min_score or c["bm25_rank"] is not None

        sources = [Source(file=_source_file(c["metadata"]), snippet=c["text"][:200],
                          score=round(c["dense_score"], 3)) for _, c in fused]
        context = "\n\n---\n\n".join(
            f"[{_source_file(c['metadata'])}]\n{c['text']}" for _, c in fused)
        has_strong = any(strong(c) for _, c in fused) if fused else False
        memories = get_storage(self._settings).memories.recall(question)
        if memories:
            memory_text = "\n".join(f"- ({item['kind']}) {item['text']}" for item in memories)
            context = (context + "\n\n---\n\n" if context else "") + \
                f"[Personal memory]\n{memory_text}"
            query_terms = set(_tokenize(question))
            for item in memories:
                memory_terms = set(_tokenize(item["text"]))
                relevant = item["kind"] != "preference" or bool(query_terms & memory_terms)
                if relevant:
                    sources.append(Source(file="personal memory",
                                          snippet=item["text"][:200], score=1.0))
                    has_strong = True
        return context, sources, has_strong

    def query(self, question: str, top_k: int | None = None,
              deep_think: bool = False,
              history: list[dict] | None = None) -> QueryResult:
        t0 = time.perf_counter()
        if persona.is_self_question(question):
            return QueryResult(answer=persona.self_description(self._settings.llm_model), sources=[],
                               timing={"retrieve_ms": 0,
                                       "generate_ms": round((time.perf_counter() - t0) * 1000),
                                       "tokens": 0, "tok_per_s": 0.0})
        reminder_answer = persona.reminder_limit_response(question)
        if reminder_answer is not None:
            return QueryResult(answer=reminder_answer, sources=[],
                               timing={"retrieve_ms": 0, "generate_ms": 0,
                                       "tokens": 0, "tok_per_s": 0.0})
        remembered = parse_remember_command(question)
        if remembered:
            get_storage(self._settings).memories.add(remembered)
            return QueryResult(answer="Got it — I'll remember that.", sources=[],
                               timing={"retrieve_ms": 0, "generate_ms": 0,
                                       "tokens": 0, "tok_per_s": 0.0})
        action_answer = run_browser_command(question, self._browser_opener)
        if action_answer is not None:
            return QueryResult(answer=action_answer, sources=[],
                               timing={"retrieve_ms": 0, "generate_ms": 0,
                                       "tokens": 0, "tok_per_s": 0.0})
        scheduled_answer = appointment_schedule(question)
        if scheduled_answer is not None:
            return QueryResult(answer=scheduled_answer, sources=[],
                               timing={"retrieve_ms": 0, "generate_ms": 0,
                                       "tokens": 0, "tok_per_s": 0.0})
        k = top_k or self._settings.top_k
        personal_question = _needs_personal_retrieval(question)
        if personal_question:
            search_question = _retrieval_question(question, history)
            try:
                context, sources, has_strong = self._retrieve(search_question, k)
            except (ConnectionError, TimeoutError) as e:
                # LlamaIndex's Ollama embedding adapter uses the Ollama SDK, whose
                # connection failures are not translated into OllamaDown for us.
                raise OllamaDown(
                    f"Ollama unreachable at {self._settings.ollama_host}: {e}") from e
        else:
            context, sources, has_strong = "", [], False
        t1 = time.perf_counter()
        if not has_strong and personal_question:
            return QueryResult(answer=persona.REFUSAL, sources=sources,
                               timing={"retrieve_ms": round((t1 - t0) * 1000),
                                       "generate_ms": 0, "tokens": 0, "tok_per_s": 0.0})
        if not has_strong:
            sources = []
        messages = [
            {"role": "system", "content": persona.build_system_prompt(
                context if has_strong else "", grounded_only=personal_question)},
            *(history or []),
            {"role": "user", "content": question},
        ]
        res = self._client.chat(
            messages, think=should_use_deep_thinking(
                question, deep_think, self._settings.llm_model),
            max_tokens=self._settings.deep_think_max_tokens
            if should_use_deep_thinking(
                question, deep_think, self._settings.llm_model) else None,
            keep_alive=self._settings.keep_alive)
        t2 = time.perf_counter()
        timing = {"retrieve_ms": round((t1 - t0) * 1000), "generate_ms": round((t2 - t1) * 1000),
                  "tokens": res.eval_count, "tok_per_s": res.tok_per_s}
        return QueryResult(answer=res.text, sources=sources, timing=timing)

    def query_stream(self, question: str, top_k: int | None = None,
                     deep_think: bool = False,
                     history: list[dict] | None = None) -> Iterator[dict]:
        t0 = time.perf_counter()
        if persona.is_self_question(question):
            yield {"type": "delta", "text": persona.self_description(self._settings.llm_model)}
            yield {"type": "final", "sources": [],
                   "timing": {"retrieve_ms": round((time.perf_counter() - t0) * 1000)}}
            return
        reminder_answer = persona.reminder_limit_response(question)
        if reminder_answer is not None:
            yield {"type": "delta", "text": reminder_answer}
            yield {"type": "final", "sources": [], "timing": {"retrieve_ms": 0}}
            return
        remembered = parse_remember_command(question)
        if remembered:
            get_storage(self._settings).memories.add(remembered)
            yield {"type": "delta", "text": "Got it — I'll remember that."}
            yield {"type": "final", "sources": [], "timing": {"retrieve_ms": 0}}
            return
        action_answer = run_browser_command(question, self._browser_opener)
        if action_answer is not None:
            yield {"type": "delta", "text": action_answer}
            yield {"type": "final", "sources": [],
                   "timing": {"retrieve_ms": 0, "generate_ms": 0,
                              "tokens": 0, "tok_per_s": 0.0}}
            return
        scheduled_answer = appointment_schedule(question)
        if scheduled_answer is not None:
            yield {"type": "delta", "text": scheduled_answer}
            yield {"type": "final", "sources": [],
                   "timing": {"retrieve_ms": 0, "generate_ms": 0,
                              "tokens": 0, "tok_per_s": 0.0}}
            return
        k = top_k or self._settings.top_k
        personal_question = _needs_personal_retrieval(question)
        if personal_question:
            search_question = _retrieval_question(question, history)
            try:
                context, sources, has_strong = self._retrieve(search_question, k)
            except (ConnectionError, TimeoutError) as e:
                raise OllamaDown(
                    f"Ollama unreachable at {self._settings.ollama_host}: {e}") from e
        else:
            context, sources, has_strong = "", [], False
        t1 = time.perf_counter()
        first = {"retrieve_ms": round((t1 - t0) * 1000)}
        if not has_strong and personal_question:
            yield {"type": "delta", "text": persona.REFUSAL}
            yield {"type": "final", "sources": [asdict(s) for s in sources], "timing": first}
            return
        if not has_strong:
            sources = []
        messages = [
            {"role": "system", "content": persona.build_system_prompt(
                context if has_strong else "", grounded_only=personal_question)},
            *(history or []),
            {"role": "user", "content": question},
        ]
        for delta in self._client.chat_stream(
                messages, think=should_use_deep_thinking(
                    question, deep_think, self._settings.llm_model),
                max_tokens=self._settings.deep_think_max_tokens
                if should_use_deep_thinking(
                    question, deep_think, self._settings.llm_model) else None,
                keep_alive=self._settings.keep_alive):
            yield {"type": "delta", "text": delta}
        yield {"type": "final", "sources": [asdict(s) for s in sources], "timing": first}

    def chunk_count(self) -> int:
        if self._qdrant is not None:
            return self._qdrant.count(COLLECTION, exact=True).count
        if not self._settings.index_path.exists():
            return 0
        from qdrant_client import QdrantClient

        c = QdrantClient(path=str(self._settings.index_path))
        try:
            return c.count(COLLECTION, exact=True).count
        finally:
            c.close()


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
