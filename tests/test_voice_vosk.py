import io
import wave
from pathlib import Path

import pytest

from brain.config import Settings
from brain.vosk_stt import VoskEngine
from brain.voice import VoskError


def wav_bytes(rate=16000, channels=1, sampwidth=2, frames=b"\x00\x00" * 100) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


def make_model_dir(tmp_path, name="vosk-model-small-en-us-0.15") -> Path:
    d = tmp_path / "vosk" / name
    d.mkdir(parents=True)
    (d / "am").write_bytes(b"model")  # any marker file
    return d


class FakeRecognizer:
    def __init__(self, seen, rate_log):
        self.seen = seen
        self.rate_log = rate_log
        self._calls = 0

    def AcceptWaveForm(self, data: bytes) -> bool:
        self.seen.append(data)
        self._calls += 1
        return self._calls == 1

    def FinalResult(self) -> dict:
        return {"text": "hello world",
                "result": [{"word": "hello", "conf": 0.9},
                           {"word": "world", "conf": 0.7}]}


def fake_factory(seen, rate_log):
    def factory(model_dir: Path, rate: int):
        rate_log.append((str(model_dir), rate))
        return FakeRecognizer(seen, rate_log)
    return factory


def test_discover_uses_model_under_root(tmp_path):
    model = make_model_dir(tmp_path)
    eng = VoskEngine.discover(Settings(vosk_model="vosk-model-small-en-us-0.15"),
                              models_root=tmp_path / "vosk")
    assert eng.model_dir == model


def test_discover_prefers_absolute_path(tmp_path):
    custom = tmp_path / "elsewhere" / "m"
    custom.mkdir(parents=True)
    eng = VoskEngine.discover(Settings(vosk_model=str(custom)),
                              models_root=tmp_path / "vosk")
    assert eng.model_dir == custom


def test_discover_missing_model_names_setup_command(tmp_path):
    with pytest.raises(VoskError, match="setup_voice"):
        VoskEngine.discover(Settings(vosk_model="nope"), models_root=tmp_path / "vosk")


def test_transcribe_returns_text_and_confidence(tmp_path):
    seen, rate_log = [], []
    eng = VoskEngine(make_model_dir(tmp_path), recognizer_factory=fake_factory(seen, rate_log))
    t = eng.transcribe(wav_bytes())
    assert t.text == "hello world"
    assert t.confidence == pytest.approx(0.8)
    assert b"".join(seen) == b"\x00\x00" * 100  # mono 16-bit frames reach the recognizer
    assert rate_log[0][1] == 16000


def test_transcribe_downmixes_stereo_to_mono(tmp_path):
    seen, rate_log = [], []
    eng = VoskEngine(make_model_dir(tmp_path), recognizer_factory=fake_factory(seen, rate_log))
    # Each stereo frame contains two 16-bit samples (4 bytes), so 100 frames
    # should produce 100 mono samples (200 bytes).
    stereo = wav_bytes(channels=2, frames=b"\x11\x22\x33\x44" * 100)
    eng.transcribe(stereo)
    assert len(b"".join(seen)) == 200  # 100 mono samples, not 200


def test_transcribe_undecodable_wav_is_loud(tmp_path):
    eng = VoskEngine(make_model_dir(tmp_path), recognizer_factory=fake_factory([], []))
    with pytest.raises(VoskError, match="WAV"):
        eng.transcribe(b"this is not a wav file at all")


def test_transcribe_silence_returns_empty_transcript(tmp_path):
    class Silent:
        def AcceptWaveForm(self, data):
            return False
        def FinalResult(self):
            return {"text": "", "result": []}

    eng = VoskEngine(make_model_dir(tmp_path),
                     recognizer_factory=lambda d, r: Silent())
    t = eng.transcribe(wav_bytes())
    assert t.text == ""
    assert t.confidence == 0.0


def test_model_loads_lazily_exactly_once(tmp_path, monkeypatch):
    # The expensive vosk.Model must load once and be cached behind the lock;
    # a KaldiRecognizer is (correctly) created per transcribe.
    import vosk

    model_loads = []

    class FakeModel:
        def __init__(self, path):
            model_loads.append(path)

    class FakeKaldi:
        def __init__(self, model, rate):
            assert isinstance(model, FakeModel)

        def AcceptWaveForm(self, data):
            return False

        def FinalResult(self):
            return {"text": "ok", "result": []}

    monkeypatch.setattr(vosk, "Model", FakeModel)
    monkeypatch.setattr(vosk, "KaldiRecognizer", FakeKaldi)
    eng = VoskEngine(make_model_dir(tmp_path))
    eng.transcribe(wav_bytes())
    eng.transcribe(wav_bytes())
    assert len(model_loads) == 1
