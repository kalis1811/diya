"""Fine-tune a small Qwen model with a local chat dataset using 4-bit LoRA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


def validate_jsonl(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise SystemExit(f"Dataset file not found: {path}")
    count = 0
    eligible = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise SystemExit(f"Missing messages list at {path}:{line_number}")
            if (messages[-1].get("role") != "assistant"
                    or len(messages) < 2
                    or messages[-2].get("role") != "user"):
                raise SystemExit(
                    f"Example must end with an incoming message followed by your reply "
                    f"at {path}:{line_number}"
                )
            if any(message.get("role") not in {"user", "assistant"}
                   or not isinstance(message.get("content"), str)
                   for message in messages):
                raise SystemExit(f"Invalid message format at {path}:{line_number}")
            count += 1
            if len(messages) >= 2 and messages[-2].get("role") == "user":
                eligible += 1
    if count == 0:
        raise SystemExit(f"Dataset is empty: {path}")
    print(f"Validated {count} examples in {path}; {eligible} have incoming chat context")
    if eligible == 0:
        raise SystemExit(f"No examples with incoming chat context in {path}")
    return count, eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Hugging Face model ID or local model directory")
    parser.add_argument("--train", type=Path, default=ROOT / "data/finetune/train.jsonl")
    parser.add_argument("--validation", type=Path,
                        default=ROOT / "data/finetune/validation.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/finetune/qwen3.5-0.8b-lora-v3")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected. Install CUDA-enabled PyTorch first.")
    if args.max_length < 32:
        raise SystemExit("--max-length must be at least 32")
    validate_jsonl(args.train)
    validate_jsonl(args.validation)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(args.model)

    datasets = load_dataset(
        "json",
        data_files={"train": str(args.train), "validation": str(args.validation)},
    )
    for split in ("train", "validation"):
        original_count = len(datasets[split])
        datasets[split] = datasets[split].filter(
            lambda row: len(row["messages"]) >= 2
            and row["messages"][-2].get("role") == "user"
        )
        dropped = original_count - len(datasets[split])
        if dropped:
            print(f"Skipping {dropped} {split} examples without incoming chat context")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    training = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-4,
        warmup_steps=2,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=True,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(
        model=model,
        args=training,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=processor,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    processor.save_pretrained(str(args.output))
    print(f"Adapter saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
