import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from brain.voice import PiperError

Runner = Callable[[list[str], bytes], tuple[int, bytes]]

SETUP_HINT = "run: uv run python scripts/setup_voice.py"


def _default_runner(cmd: list[str], stdin_bytes: bytes) -> tuple[int, bytes]:
    proc = subprocess.run(cmd, input=stdin_bytes, capture_output=True)
    return proc.returncode, proc.stderr


class PiperEngine:
    def __init__(self, exe: Path, voice: Path, runner: Runner | None = None):
        self.exe = Path(exe)
        self.voice = Path(voice)
        self._run = runner or _default_runner

    @classmethod
    def discover(cls, settings, *, tools_dir: Path | None = None,
                 models_dir: Path | None = None) -> "PiperEngine":
        from brain.config import PROJECT_ROOT
        tools = Path(tools_dir) if tools_dir else PROJECT_ROOT / "tools" / "piper"
        models = Path(models_dir) if models_dir else PROJECT_ROOT / "storage" / "models" / "piper"
        exe = Path(settings.piper_exe) if settings.piper_exe else tools / "piper.exe"
        if not exe.is_file():
            raise PiperError(
                f"Piper executable not found at {exe} - {SETUP_HINT} "
                f"(or set BRAIN_PIPER_EXE)")
        voice = cls._resolve_voice(models, settings.piper_voice)
        return cls(exe, voice)

    @staticmethod
    def _resolve_voice(models_dir: Path, preferred: str) -> Path:
        candidates = sorted(p for p in models_dir.glob("*.onnx")
                            if p.with_suffix(".onnx.json").is_file())
        exact = models_dir / f"{preferred}.onnx"
        if exact in candidates:
            return exact
        if not candidates:
            raise PiperError(
                f"No Piper voice (*.onnx with .onnx.json sidecar) found under {models_dir} "
                f"- {SETUP_HINT}")
        raise PiperError(
            f"Piper voice '{preferred}' not found under {models_dir}; "
            f"installed voices: {', '.join(c.name for c in candidates)}")

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise PiperError("refusing to synthesize empty text")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = Path(tmp.name)
        try:
            cmd = [str(self.exe), "-m", str(self.voice), "-f", str(out_path)]
            try:
                code, stderr = self._run(cmd, text.encode("utf-8"))
            except OSError as e:
                raise PiperError(f"could not start Piper: {e} - {SETUP_HINT}") from e
            if code != 0:
                detail = stderr.decode(errors="replace").strip()[:300]
                raise PiperError(
                    f"piper exited {code}: {detail or 'no stderr'} - "
                    f"{SETUP_HINT} if the binary or voice is broken")
            if not out_path.is_file():
                raise PiperError("piper reported success but produced no WAV file")
            return out_path.read_bytes()
        finally:
            out_path.unlink(missing_ok=True)
