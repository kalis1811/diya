"""Merge a Phi-4 Mini PEFT adapter into a local base model for Ollama import."""
from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path,
                        help="local Phi-4 Mini base model directory")
    parser.add_argument("--adapter", required=True, type=Path,
                        help="saved PEFT adapter directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="new directory for the merged model")
    args = parser.parse_args()

    if not args.model.is_dir():
        raise SystemExit(f"Base model directory not found: {args.model}")
    if not args.adapter.is_dir():
        raise SystemExit(f"Adapter directory not found: {args.adapter}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")
    if not (args.adapter / "adapter_config.json").is_file():
        raise SystemExit(f"PEFT adapter config not found in {args.adapter}")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading Phi base model from: {args.model}")
    base = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    print(f"Loading LoRA adapter from: {args.adapter}")
    tuned = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    print("Merging adapter weights...")
    merged = tuned.merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    tokenizer.save_pretrained(args.output)
    print(f"Merged model saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
