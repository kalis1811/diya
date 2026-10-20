"""Compare a saved Qwen LoRA adapter with its base model on held-out chats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]


def read_examples(path: Path, count: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            messages = row.get("messages", [])
            if (len(messages) >= 2
                    and messages[-1].get("role") == "assistant"
                    and messages[-2].get("role") == "user"):
                rows.append(row)
                if len(rows) >= count:
                    break
    if not rows:
        raise SystemExit(f"No suitable validation examples found in {path}")
    return rows


def answer(model, processor, messages: list[dict], max_new_tokens: int) -> str:
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_length = encoded["input_ids"].shape[1]
    return processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True
    )[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter", type=Path,
                        default=ROOT / "data/finetune/qwen3.5-0.8b-lora-v3")
    parser.add_argument("--validation", type=Path,
                        default=ROOT / "data/finetune/validation.jsonl")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected. Activate the diya-train environment.")
    if not args.adapter.is_dir():
        raise SystemExit(f"Adapter folder not found: {args.adapter}")
    rows = read_examples(args.validation, args.count)
    processor = AutoProcessor.from_pretrained(args.model)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    base.eval()

    base_answers = []
    for row in rows:
        prompt = row["messages"][:-1]
        base_answers.append(answer(base, processor, prompt, args.max_new_tokens))

    tuned = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    tuned.eval()
    for index, (row, base_text) in enumerate(zip(rows, base_answers), 1):
        prompt = row["messages"][:-1]
        tuned_text = answer(tuned, processor, prompt, args.max_new_tokens)
        print(f"\n=== Validation example {index} ===")
        print("Expected reply:", row["messages"][-1]["content"])
        print("Base model:   ", base_text)
        print("With adapter: ", tuned_text)


if __name__ == "__main__":
    main()
