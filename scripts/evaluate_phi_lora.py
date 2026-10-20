"""Compare Phi-4 Mini with and without Diya's caretaker LoRA adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def answer(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = encoded["input_ids"].shape[1]
    return tokenizer.decode(
        generated[0, prompt_length:], skip_special_tokens=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Hugging Face model ID or local model directory")
    parser.add_argument("--adapter", type=Path,
                        default=ROOT / "data/finetune/phi4-mini-caretaker-lora-v1")
    parser.add_argument("--validation", type=Path,
                        default=ROOT / "data/finetune/caretaker_validation.jsonl")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected. Activate the diya-train environment.")
    if not args.adapter.is_dir():
        raise SystemExit(f"Adapter folder not found: {args.adapter}")
    rows = read_examples(args.validation, args.count)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    base.eval()
    base_answers = [
        answer(base, tokenizer, row["messages"][:-1], args.max_new_tokens)
        for row in rows
    ]

    tuned = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    tuned.eval()
    for index, (row, base_text) in enumerate(zip(rows, base_answers), 1):
        tuned_text = answer(tuned, tokenizer, row["messages"][:-1], args.max_new_tokens)
        print(f"\n=== Validation example {index} ===")
        print("Expected reply:", row["messages"][-1]["content"])
        print("Base model:   ", base_text)
        print("With adapter: ", tuned_text)


if __name__ == "__main__":
    main()
