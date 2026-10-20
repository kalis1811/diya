import io
import struct
import threading
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

from brain.voice import VoskError

SETUP_HINT = "run: uv run python scripts/setup_voice.py"

_CHUNK_FRAMES = 4000  # VENOM-proven Kaldi chunk size


@dataclass
class Transcript:
    text: str
    confidence: float


class VoskEngine:
    def __init__(self, model_dir: Path, recognizer_factory=None):
        self.model_dir = Path(model_dir)
        self._factory = recognizer_factory
        self._model = None
        self._lock = threading.Lock()

    @classmethod
    def discover(cls, settings, *, models_root: Path | None = None) -> "VoskEngine":
        from brain.config import PROJECT_ROOT
        root = Path(models_root) if models_root else PROJECT_ROOT / "storage" / "models" / "vosk"
        candidate = Path(settings.vosk_model)
        model_dir = candidate if candidate.is_absolute() else root / settings.vosk_model
        if not model_dir.is_dir():
            raise VoskError(
                f"Vosk model not found at {model_dir} - {SETUP_HINT} "
                f"(or set BRAIN_VOSK_MODEL to an absolute path)")
        return cls(model_dir)

    def _make_recognizer(self, rate: int):
        if self._factory is not None:
            return self._factory(self.model_dir, rate)
        with self._lock:
            if self._model is None:
                from vosk import Model, SetLogLevel
                SetLogLevel(-1)
                self._model = Model(str(self.model_dir))
            from vosk import KaldiRecognizer
            return KaldiRecognizer(self._model, rate)

    def transcribe(self, wav_bytes: bytes) -> Transcript:
        raw, channels, rate, sampwidth = self._decode_wav(wav_bytes)
        if sampwidth != 2:
            raise VoskError(
                f"unsupported WAV sample width {sampwidth} bytes - record 16-bit PCM")
        mono = self._downmix(raw, channels)
        rec = self._make_recognizer(rate)
        chunk = _CHUNK_FRAMES * 2
        for i in range(0, len(mono), chunk):
            rec.AcceptWaveForm(mono[i:i + chunk])
        final = rec.FinalResult()
        text = (final.get("text") or "").strip()
        confs = [w["conf"] for w in (final.get("result") or []) if "conf" in w]
        confidence = round(sum(confs) / len(confs), 3) if confs else 0.0
        return Transcript(text=text, confidence=confidence)

    @staticmethod
    def _decode_wav(wav_bytes: bytes) -> tuple[bytes, int, int, int]:
        try:
            with wave.open(io.BytesIO(wav_bytes)) as w:
                channels = w.getnchannels()
                rate = w.getframerate()
                sampwidth = w.getsampwidth()
                raw = w.readframes(w.getnframes())
        except (wave.Error, struct.error, EOFError) as e:
            raise VoskError(f"not a decodable WAV file: {e}") from e
        return raw, channels, rate, sampwidth

    @staticmethod
    def _downmix(raw: bytes, channels: int) -> bytes:
        if channels == 1:
            return raw
        samples = array("h")
        samples.frombytes(raw)
        mono = array("h", (sum(samples[i:i + channels]) // channels
                           for i in range(0, len(samples), channels)))
        return mono.tobytes()
