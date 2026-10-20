"""Install local Piper and Vosk assets from a working Project1 voice setup."""
import argparse
import shutil
from pathlib import Path

from brain.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy Piper voice files into this project")
    parser.add_argument("--from", dest="source", type=Path, required=True,
                        help="source project containing tools/piper and storage/models/piper")
    args = parser.parse_args()
    source_tools = args.source / "tools" / "piper"
    source_model = args.source / "storage" / "models" / "piper"
    source_vosk = (
        args.source / "storage" / "models" / "vosk"
        / "vosk-model-small-en-us-0.15"
    )
    if not (source_tools / "piper.exe").is_file():
        parser.error(f"Piper executable not found under {source_tools}")
    models = sorted(source_model.glob("en_US-amy-medium.onnx"))
    model = models[0] if models else None
    if model is None or not model.with_suffix(".onnx.json").is_file():
        parser.error(f"Amy voice model and JSON sidecar not found under {source_model}")
    if not all((source_vosk / name).exists() for name in ("am", "conf", "graph")):
        parser.error(f"Vosk model files not found under {source_vosk}")

    target_tools = PROJECT_ROOT / "tools" / "piper"
    target_models = PROJECT_ROOT / "storage" / "models" / "piper"
    target_vosk = PROJECT_ROOT / "storage" / "models" / "vosk" / source_vosk.name
    shutil.copytree(source_tools, target_tools, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.wav"))
    target_models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, target_models / model.name)
    shutil.copy2(model.with_suffix(".onnx.json"), target_models / model.with_suffix(".onnx.json").name)
    shutil.copytree(source_vosk, target_vosk, dirs_exist_ok=True)
    for path in (target_tools / "piper.exe", target_models / model.name,
                 target_models / model.with_suffix(".onnx.json").name,
                 target_vosk / "conf" / "model.conf"):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Voice setup did not produce a valid file: {path}")
        print(f"Ready: {path} ({path.stat().st_size:,} bytes)")
    print(f"Ready: {target_vosk} (Vosk speech recognition model)")


if __name__ == "__main__":
    main()
