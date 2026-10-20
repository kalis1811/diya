import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DEFAULTS = {
    "BRAIN_OLLAMA_HOST": "http://localhost:11434",
    "BRAIN_LLM_MODEL": "phi4-mini:latest",
    "BRAIN_EMBED_MODEL": "qwen3-embedding:0.6b",
    "BRAIN_CHUNK_SIZE": "512",
    "BRAIN_CHUNK_OVERLAP": "50",
    "BRAIN_TOP_K": "5",
    "BRAIN_API_PORT": "8400",
    "BRAIN_KEEP_ALIVE": "30m",
    "BRAIN_DEEP_THINK_MAX_TOKENS": "3072",
    "BRAIN_MIN_SCORE": "0.25",
    "BRAIN_TTS_ENGINE": "auto",
    "BRAIN_STT_ENGINE": "auto",
    "BRAIN_PIPER_EXE": "",
    "BRAIN_PIPER_VOICE": "en_US-amy-medium",
    "BRAIN_VOSK_MODEL": "vosk-model-small-en-us-0.15",
    "BRAIN_MONGODB_URI": "mongodb://127.0.0.1:27017",
    "BRAIN_MONGODB_DATABASE": "diya",
    "BRAIN_ADMIN_USERNAME": "admin",
    "BRAIN_ADMIN_PASSWORD": "",
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
    tts_engine: str = _DEFAULTS["BRAIN_TTS_ENGINE"]
    stt_engine: str = _DEFAULTS["BRAIN_STT_ENGINE"]
    piper_exe: str = _DEFAULTS["BRAIN_PIPER_EXE"]
    piper_voice: str = _DEFAULTS["BRAIN_PIPER_VOICE"]
    vosk_model: str = _DEFAULTS["BRAIN_VOSK_MODEL"]
    mongodb_uri: str = _DEFAULTS["BRAIN_MONGODB_URI"]
    mongodb_database: str = _DEFAULTS["BRAIN_MONGODB_DATABASE"]
    admin_username: str = _DEFAULTS["BRAIN_ADMIN_USERNAME"]
    admin_password: str = _DEFAULTS["BRAIN_ADMIN_PASSWORD"]

    @property
    def raw_dirs(self) -> dict[str, Path]:
        return {cat: self.raw_dir / cat for cat in ("docs", "web", "chats")}

    @property
    def memory_path(self) -> Path:
        return self.raw_dir.parent / "memory.json"


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
        tts_engine=os.environ.get("BRAIN_TTS_ENGINE", _DEFAULTS["BRAIN_TTS_ENGINE"]),
        stt_engine=os.environ.get("BRAIN_STT_ENGINE", _DEFAULTS["BRAIN_STT_ENGINE"]),
        piper_exe=os.environ.get("BRAIN_PIPER_EXE", _DEFAULTS["BRAIN_PIPER_EXE"]),
        piper_voice=os.environ.get("BRAIN_PIPER_VOICE", _DEFAULTS["BRAIN_PIPER_VOICE"]),
        vosk_model=os.environ.get("BRAIN_VOSK_MODEL", _DEFAULTS["BRAIN_VOSK_MODEL"]),
        mongodb_uri=os.environ.get("BRAIN_MONGODB_URI", _DEFAULTS["BRAIN_MONGODB_URI"]),
        mongodb_database=os.environ.get("BRAIN_MONGODB_DATABASE", _DEFAULTS["BRAIN_MONGODB_DATABASE"]),
        admin_username=os.environ.get("BRAIN_ADMIN_USERNAME", _DEFAULTS["BRAIN_ADMIN_USERNAME"]),
        admin_password=os.environ.get("BRAIN_ADMIN_PASSWORD", _DEFAULTS["BRAIN_ADMIN_PASSWORD"]),
    )
