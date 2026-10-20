"""Merge a local PEFT LoRA adapter into its base model for deployment."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path,
                        help="local base model directory")
    parser.add_argument("--adapter", required=True, type=Path,
                        help="saved LoRA adapter directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="new directory for the merged model")
    args = parser.parse_args()

    if not args.model.is_dir():
        raise SystemExit(f"Base model directory not found: {args.model}")
    if not args.adapter.is_dir():
        raise SystemExit(f"Adapter directory not found: {args.adapter}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading base model from: {args.model}")
    base = AutoModelForImageTextToText.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    print(f"Loading adapter from: {args.adapter}")
    tuned = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    merged = tuned.merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    processor.save_pretrained(args.output)
    print(f"Merged model saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
