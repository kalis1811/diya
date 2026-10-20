import dataclasses
from pathlib import Path

from brain.config import Settings, get_settings


def test_defaults_match_spec():
    s = Settings()
    assert s.ollama_host == "http://localhost:11434"
    assert s.llm_model == "phi4-mini:latest"
    assert s.embed_model == "qwen3-embedding:0.6b"
    assert s.chunk_size == 512
    assert s.chunk_overlap == 50
    assert s.top_k == 5
    assert s.api_port == 8400
    assert s.keep_alive == "30m"
    assert s.deep_think_max_tokens == 3072
    assert s.min_score == 0.25


def test_voice_defaults_match_spec():
    s = Settings()
    assert s.tts_engine == "auto"
    assert s.stt_engine == "auto"
    assert s.piper_exe == ""  # empty = auto-discover under tools/piper
    assert s.piper_voice == "en_US-amy-medium"
    assert s.vosk_model == "vosk-model-small-en-us-0.15"


def test_voice_env_override(monkeypatch):
    monkeypatch.setenv("BRAIN_TTS_ENGINE", "off")
    monkeypatch.setenv("BRAIN_STT_ENGINE", "off")
    monkeypatch.setenv("BRAIN_PIPER_EXE", "D:/elsewhere/piper.exe")
    monkeypatch.setenv("BRAIN_PIPER_VOICE", "en_US-lessac-medium")
    monkeypatch.setenv("BRAIN_VOSK_MODEL", "vosk-model-en-us-0.22")
    get_settings.cache_clear()
    s = get_settings()
    assert s.tts_engine == "off"
    assert s.stt_engine == "off"
    assert s.piper_exe == "D:/elsewhere/piper.exe"
    assert s.piper_voice == "en_US-lessac-medium"
    assert s.vosk_model == "vosk-model-en-us-0.22"
    get_settings.cache_clear()  # don't leak into other tests


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
