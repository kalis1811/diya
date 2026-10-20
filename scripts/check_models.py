"""Check the configured local Ollama models before starting the app."""
from __future__ import annotations

import shutil
import subprocess
import sys

from brain.config import get_settings


def main() -> int:
    executable = shutil.which("ollama")
    if executable is None:
        print("Ollama was not found. Install Ollama and start its local service.")
        return 1

    try:
        result = subprocess.run(
            [executable, "list"], capture_output=True, text=True, timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Could not contact Ollama: {exc}. Start Ollama and try again.")
        return 1
    if result.returncode != 0:
        print("Ollama is not responding. Start the Ollama app or service and try again.")
        if result.stderr.strip():
            print(result.stderr.strip())
        return 1

    installed = {
        line.split()[0] for line in result.stdout.splitlines()[1:] if line.split()
    }
    settings = get_settings()
    required = (settings.llm_model, settings.embed_model)
    missing = [model for model in required if model not in installed]
    if missing:
        print("The following configured Ollama model(s) are missing:")
        for model in missing:
            print(f"  ollama pull {model}")
        print("Run those commands, then start Diya again.")
        return 1

    print(f"Ollama is ready with {settings.llm_model} and {settings.embed_model}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
