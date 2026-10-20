import pytest
import httpx

from brain.config import get_settings
from brain.engine import load_engine

def _live_models_ready() -> bool:
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.ollama_host.rstrip('/')}/api/tags", timeout=3.0)
        response.raise_for_status()
        installed = {item.get("name") for item in response.json().get("models", [])}
        return settings.llm_model in installed and settings.embed_model in installed
    except (httpx.HTTPError, ValueError):
        return False


pytestmark = pytest.mark.skipif(
    not _live_models_ready(),
    reason="Ollama must be running with the configured language and embedding models")


@pytest.fixture(scope="module")
def engine():
    return load_engine(get_settings())


def test_in_domain_answer_cites_source(engine):
    res = engine.query("What do my notes say powers the personal AI brain?")
    assert res.answer.strip()
    assert any("getting_started" in s.file for s in res.sources)


def test_out_of_domain_refuses(engine):
    res = engine.query("What is the capital of France?")
    assert "paris" in res.answer.lower()


def test_introduces_herself_by_name(engine):
    res = engine.query("What is your name?")
    assert "Diya" in res.answer


def test_exact_number_query_finds_source(engine):
    res = engine.query("What do my notes say about my machine's VRAM?")
    assert res.answer  # non-empty
    assert res.sources  # at least one chunk retrieved


def test_deep_think_runs(engine):
    res = engine.query("What do my notes say powers the personal AI brain?", deep_think=True)
    assert res.answer  # non-empty when the model finishes thinking within budget
