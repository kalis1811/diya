from pathlib import Path

from brain.config import Settings
from brain.engine import BrainEngine, EmptyIndexError, QueryResult, Source
from brain.ollama_client import ChatResult, OllamaDown


class FakeNode:
    def __init__(self, id_, text, score, meta):
        self.id_, self.text, self.score, self.metadata = id_, text, score, meta

    def get_content(self):
        return self.text


class FakeRetriever:
    def __init__(self, nodes, calls):
        self.nodes = nodes
        self.calls = calls

    def retrieve(self, q):
        self.calls.append(q)
        return self.nodes


class FakeIndex:
    def __init__(self, nodes):
        self.nodes = nodes
        self.retrieve_calls = []

    def as_retriever(self, similarity_top_k=5):
        return FakeRetriever(self.nodes, self.retrieve_calls)


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
    res = eng.query("According to my notes, when is the launch?")
    assert isinstance(res, QueryResult)
    assert "2026-11-01" in res.answer
    assert [src.file for src in res.sources] == ["meeting.md"]
    assert res.sources[0].score == 0.82
    assert set(res.timing) == {"retrieve_ms", "generate_ms", "tokens", "tok_per_s"}


def test_query_default_uses_no_thinking(tmp_path):
    s = make_settings(tmp_path)
    client = FakeClient()
    eng = make_engine(s, client)
    eng.query("What did I tell you?")
    call = client.calls[0]
    assert call["think"] is False
    assert call["max_tokens"] is None
    assert call["keep_alive"] == s.keep_alive


def test_general_support_request_does_not_search_personal_files(tmp_path):
    client = FakeClient()
    eng = make_engine(make_settings(tmp_path), client)
    result = eng.query("I had a hard day and don't want advice. Please listen.")
    assert result.sources == []
    assert eng._index.retrieve_calls == []
    assert "Mass Nithish" not in client.calls[0]["messages"][0]["content"]


def test_self_contained_schedule_uses_current_request_not_chat_templates(tmp_path):
    client = FakeClient()
    eng = make_engine(make_settings(tmp_path), client)
    result = eng.query(
        "I'm overwhelmed. It's 1 PM, my appointment is at 3 PM, and travel takes "
        "30 minutes one way. I also need groceries, but don't know how long shopping "
        "takes. Make a short schedule. Don't invent times or promise reminders.")
    assert result.sources == []
    assert eng._index.retrieve_calls == []
    assert client.calls == []
    assert "2:15 PM: Leave" in result.answer
    assert "2:45 PM: Arrive" in result.answer
    assert "After the appointment" in result.answer
    assert "12:30" not in result.answer


def test_schedule_asks_for_missing_travel_time_without_searching_chats(tmp_path):
    client = FakeClient()
    eng = make_engine(make_settings(tmp_path), client)
    result = eng.query(
        "It's 1 PM, my appointment is at 3 PM, and I need groceries. Make a short "
        "schedule, but I don't know how long travel takes.")
    assert result.answer == "How long does it take to travel one way to the appointment?"
    assert result.sources == []
    assert eng._index.retrieve_calls == []
    assert client.calls == []


def test_deep_think_caps_tokens(tmp_path):
    s = Settings(llm_model="qwen3.5:4b", raw_dir=tmp_path,
                 index_path=tmp_path / "idx")
    client = FakeClient()
    eng = make_engine(s, client)
    eng.query("hard question", deep_think=True)
    call = client.calls[0]
    assert call["think"] is True
    assert call["max_tokens"] == s.deep_think_max_tokens


def test_deep_think_is_disabled_for_phi_mini(tmp_path):
    s = make_settings(tmp_path)
    client = FakeClient()
    make_engine(s, client).query("hard question", deep_think=True)
    call = client.calls[0]
    assert call["think"] is False
    assert call["max_tokens"] is None


def test_system_prompt_contains_context_and_rules(tmp_path):
    s = make_settings(tmp_path)
    client = FakeClient()
    eng = make_engine(s, client)
    eng.query("What did I tell you?")
    sys_msg = client.calls[0]["messages"][0]
    assert sys_msg["role"] == "system"
    assert "meeting.md" in sys_msg["content"]      # context included
    assert "Answer only from the CONTEXT below" in sys_msg["content"]


def test_insufficient_evidence_short_circuits_with_refusal(tmp_path):
    # No chunk above min_score and no BM25 hit -> canned refusal, NO generation call,
    # but nearest sources still returned so the caller can judge.
    s = Settings(raw_dir=tmp_path, index_path=tmp_path / "idx", min_score=0.9)
    client = FakeClient()
    nodes = [FakeNode("p1", "irrelevant", 0.10, {"file_name": "a.md"})]
    eng = BrainEngine(FakeIndex(nodes), client, s)
    res = eng.query("What is in my notes?")
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
    res = eng.query("What does my file say about 4471?")
    files = [src.file for src in res.sources]
    assert "invoice.txt" in files


def test_rrf_ranks_dual_channel_hits_first(tmp_path):
    s = make_settings(tmp_path)
    nodes = [FakeNode("p1", "We decided the launch date is 2026-11-01.", 0.80,
                      {"file_name": "meeting.md"}),
             FakeNode("p2", "Venue is the old library.", 0.50, {"file_name": "dual.txt"})]
    corpus = {"p2": ("Venue is the old library.", {"file_name": "dual.txt"})}
    eng = BrainEngine(FakeIndex(nodes), FakeClient(), s, corpus=corpus)
    res = eng.query("What do my notes say about the venue old library?")
    assert res.sources[0].file == "dual.txt"  # in both channels beats dense-only p1


def test_bm25_hit_counts_as_strong_evidence(tmp_path):
    # min_score is huge, but the BM25 channel matched -> generation proceeds anyway
    s = Settings(raw_dir=tmp_path, index_path=tmp_path / "idx", min_score=0.99)
    nodes = [FakeNode("p1", "unrelated filler text here", 0.10, {"file_name": "a.md"})]
    corpus = {"p2": ("Invoice 4471 total is 88 euros", {"file_name": "invoice.txt"})}
    client = FakeClient()
    eng = BrainEngine(FakeIndex(nodes), client, s, corpus=corpus)
    res = eng.query("What does my file say about 4471?")
    assert res.answer != "I don't know - that's not in your documents."
    sys_msg = client.calls[0]["messages"][0]["content"]
    assert "invoice.txt" in sys_msg


def test_query_stream(tmp_path):
    eng = make_engine(make_settings(tmp_path))
    events = list(eng.query_stream("According to my notes, when is the launch?"))
    assert events[0]["type"] == "delta"
    assert events[-1]["type"] == "final"
    assert events[-1]["sources"][0]["file"] == "meeting.md"


def test_query_stream_short_circuits_without_generation(tmp_path):
    s = Settings(raw_dir=tmp_path, index_path=tmp_path / "idx", min_score=0.9)
    client = FakeClient()
    nodes = [FakeNode("p1", "irrelevant", 0.10, {"file_name": "a.md"})]
    eng = BrainEngine(FakeIndex(nodes), client, s)
    events = list(eng.query_stream("What is in my notes?"))
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


def test_stopword_only_bm25_match_is_not_strong_evidence(tmp_path):
    # Stopword overlap ("is", "the") must not open the evidence gate -
    # the user's flowchart contract: no strong evidence -> refusal, NO LLM call.
    from brain import persona
    s = Settings(raw_dir=tmp_path, index_path=tmp_path / "idx", min_score=0.9)
    nodes = [FakeNode("p1", "irrelevant", 0.10, {"file_name": "a.md"})]
    corpus = {"p2": ("The engine is the model and the machine has an of to in on",
                     {"file_name": "b.txt"})}
    client = FakeClient()
    eng = BrainEngine(FakeIndex(nodes), client, s, corpus=corpus)
    res = eng.query("What does my file say about this?")
    assert res.answer == persona.REFUSAL
    assert client.calls == []
    assert res.timing["tokens"] == 0


def test_chunk_count_uses_stored_qdrant_client(tmp_path):
    # The engine already holds the Qdrant local client; chunk_count must use it.
    # Opening a second client on the same path fails on the local-mode file lock
    # and silently reported 0 indexed chunks (observed live on /health).
    s = make_settings(tmp_path)

    class FakeQdrant:
        def count(self, collection_name, exact=True):
            assert collection_name == "brain"
            return type("CountResult", (), {"count": 7})()

    eng = BrainEngine(FakeIndex([]), FakeClient(), s, qdrant=FakeQdrant())
    assert eng.chunk_count() == 7


def test_embedding_connection_error_becomes_ollama_down(tmp_path):
    class BrokenRetriever:
        def retrieve(self, question):
            raise ConnectionError("connection refused")

    class BrokenIndex:
        def as_retriever(self, similarity_top_k=5):
            return BrokenRetriever()

    eng = BrainEngine(BrokenIndex(), FakeClient(), make_settings(tmp_path))
    try:
        eng.query("What is in my notes?")
        raise AssertionError("expected OllamaDown")
    except OllamaDown as exc:
        assert "connection refused" in str(exc)


def test_open_youtube_command_skips_rag(tmp_path):
    opened = []
    client = FakeClient()
    eng = BrainEngine(FakeIndex([]), client, make_settings(tmp_path),
                      browser_opener=lambda url, **kw: opened.append((url, kw)) or True)
    res = eng.query("open YouTube")
    assert res.answer == "Opened YouTube in your default browser."
    assert opened == [("https://www.youtube.com/", {"new": 2})]
    assert res.sources == []
    assert client.calls == []


def test_browser_command_streams_as_action_result(tmp_path):
    opened = []
    eng = BrainEngine(FakeIndex([]), FakeClient(), make_settings(tmp_path),
                      browser_opener=lambda url, **kw: opened.append(url) or True)
    events = list(eng.query_stream("search Google for local AI brain"))
    assert opened == ["https://www.google.com/search?q=local+AI+brain"]
    assert events[0] == {"type": "delta", "text": "Opened Google search in your default browser."}
    assert events[-1]["sources"] == []


def test_identity_question_answers_with_intro_without_llm(tmp_path):
    # "Who are you?" is not a document question - it must never hit retrieval
    # or the model, and it must never be refused by the evidence gate.
    from brain import persona
    client = FakeClient()
    eng = make_engine(make_settings(tmp_path), client)
    res = eng.query("What is your name?")
    assert "Diya" in res.answer
    assert client.calls == []
    assert res.sources == []


def test_identity_question_streams_intro(tmp_path):
    from brain import persona
    client = FakeClient()
    eng = make_engine(make_settings(tmp_path), client)
    events = list(eng.query_stream("Who are you?"))
    assert events[0] == {
        "type": "delta", "text": persona.self_description(eng._settings.llm_model)}
    assert events[-1]["type"] == "final"
    assert events[-1]["sources"] == []
    assert client.calls == []
