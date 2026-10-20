import io
import wave
from pathlib import Path

import pytest

from brain.config import Settings
from brain.piper import PiperEngine
from brain.voice import PiperError


def wav_bytes(rate=22050, channels=1, sampwidth=2, frames=b"\x00\x00" * 100) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


def make_voice_dir(tmp_path, name="en_US-amy-medium") -> Path:
    d = tmp_path / "voices"
    d.mkdir(exist_ok=True)
    (d / f"{name}.onnx").write_bytes(b"onnx")
    (d / f"{name}.onnx.json").write_bytes(b"{}")
    return d


def fake_runner(code=0, stderr=b""):
    calls = []

    def run(cmd: list[str], stdin_bytes: bytes) -> tuple[int, bytes]:
        calls.append({"cmd": cmd, "stdin": stdin_bytes})
        out = Path(cmd[cmd.index("-f") + 1])
        out.write_bytes(wav_bytes())
        return code, stderr

    run.calls = calls
    return run


def make_tools_dir(tmp_path) -> Path:
    tools = tmp_path / "t" / "piper"
    tools.mkdir(parents=True, exist_ok=True)
    (tools / "piper.exe").write_bytes(b"MZ")
    return tools


def test_discover_prefers_env_exe(tmp_path):
    exe = tmp_path / "custom" / "piper.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"MZ")
    s = Settings(piper_exe=str(exe), piper_voice="en_US-amy-medium")
    eng = PiperEngine.discover(s, tools_dir=tmp_path / "tools",
                               models_dir=make_voice_dir(tmp_path))
    assert eng.exe == exe


def test_discover_falls_back_to_tools_dir(tmp_path):
    tools = tmp_path / "tools" / "piper"
    tools.mkdir(parents=True)
    (tools / "piper.exe").write_bytes(b"MZ")
    s = Settings(piper_voice="en_US-amy-medium")
    eng = PiperEngine.discover(s, tools_dir=tools, models_dir=make_voice_dir(tmp_path))
    assert eng.exe == tools / "piper.exe"


def test_discover_missing_exe_names_setup_command(tmp_path):
    with pytest.raises(PiperError, match="setup_voice"):
        PiperEngine.discover(Settings(piper_voice="en_US-amy-medium"),
                             tools_dir=tmp_path / "nope",
                             models_dir=make_voice_dir(tmp_path))


def test_discover_finds_preferred_voice(tmp_path):
    voices = make_voice_dir(tmp_path)
    eng = PiperEngine.discover(Settings(piper_voice="en_US-amy-medium"),
                               tools_dir=make_tools_dir(tmp_path), models_dir=voices)
    assert eng.voice == voices / "en_US-amy-medium.onnx"


def test_discover_skips_voice_without_sidecar(tmp_path):
    voices = make_voice_dir(tmp_path)
    (voices / "orphan.onnx").write_bytes(b"onnx")
    eng = PiperEngine.discover(Settings(piper_voice="en_US-amy-medium"),
                               tools_dir=make_tools_dir(tmp_path), models_dir=voices)
    assert eng.voice.name == "en_US-amy-medium.onnx"


def test_discover_missing_voice_lists_candidates(tmp_path):
    voices = make_voice_dir(tmp_path, "en_US-other-medium")
    with pytest.raises(PiperError, match="en_US-other-medium"):
        PiperEngine.discover(Settings(piper_voice="en_US-amy-medium"),
                             tools_dir=make_tools_dir(tmp_path), models_dir=voices)


def test_synthesize_returns_wav_bytes(tmp_path):
    run = fake_runner()
    voice = make_voice_dir(tmp_path) / "en_US-amy-medium.onnx"
    eng = PiperEngine(tmp_path / "piper.exe", voice, runner=run)
    out = eng.synthesize("hello there")
    assert out == wav_bytes()
    call = run.calls[0]
    assert call["cmd"][0] == str(tmp_path / "piper.exe")
    assert call["cmd"][call["cmd"].index("-m") + 1] == str(voice)
    assert call["stdin"] == "hello there".encode("utf-8")


def test_synthesize_nonzero_exit_is_loud(tmp_path):
    eng = PiperEngine(tmp_path / "piper.exe",
                      make_voice_dir(tmp_path) / "en_US-amy-medium.onnx",
                      runner=lambda cmd, stdin: (1, b"model load failed"))
    with pytest.raises(PiperError, match="model load failed"):
        eng.synthesize("hello")


def test_synthesize_empty_text_refused(tmp_path):
    eng = PiperEngine(tmp_path / "piper.exe",
                      make_voice_dir(tmp_path) / "en_US-amy-medium.onnx",
                      runner=fake_runner())
    with pytest.raises(PiperError, match="empty"):
        eng.synthesize("   ")
