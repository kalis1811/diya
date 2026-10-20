"""QLoRA fine-tune Phi-4-mini-instruct on Diya's reviewed caretaker examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "microsoft/Phi-4-mini-instruct"


def validate(path: Path) -> int:
    if not path.is_file():
        raise SystemExit(f"Dataset file not found: {path}")
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            messages = row.get("messages")
            if (not isinstance(messages, list) or len(messages) < 2
                    or messages[-1].get("role") != "assistant"
                    or messages[-2].get("role") != "user"):
                raise SystemExit(
                    f"Example must end with a user message followed by an assistant reply "
                    f"at {path}:{line_number}"
                )
            if any(message.get("role") not in {"user", "assistant"}
                   or not isinstance(message.get("content"), str)
                   for message in messages):
                raise SystemExit(f"Invalid message format at {path}:{line_number}")
            count += 1
    if count == 0:
        raise SystemExit(f"Dataset is empty: {path}")
    print(f"Validated {count} examples in {path}")
    return count


def prepare_prompt_completion(row: dict, tokenizer) -> dict[str, str]:
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if not full_text.startswith(prompt):
        raise ValueError("Phi chat template did not produce a matching assistant prompt prefix")
    completion = full_text[len(prompt):]
    if not completion.strip():
        raise ValueError("Chat template produced an empty assistant completion")
    return {"prompt": prompt, "completion": completion}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Hugging Face model ID or local model directory")
    parser.add_argument("--train", type=Path,
                        default=ROOT / "data/finetune/caretaker_train.jsonl")
    parser.add_argument("--validation", type=Path,
                        default=ROOT / "data/finetune/caretaker_validation.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/finetune/phi4-mini-caretaker-lora-v3")
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="optimizer updates per training batch; default 1 for small datasets")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected. Install CUDA-enabled PyTorch first.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This training recipe requires a GPU with BF16 support.")
    if args.max_length < 64:
        raise SystemExit("--max-length must be at least 64")
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be at least 1")
    train_count = validate(args.train)
    eval_count = validate(args.validation)
    print(
        f"Training Phi-4 Mini QLoRA for {args.epochs:g} epoch(s); "
        f"max length {args.max_length}; learning rate {args.learning_rate:g}; "
        f"gradient accumulation {args.gradient_accumulation_steps}"
    )
    print("Only the assistant completion is used for the loss.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_data = load_dataset("json", data_files=str(args.train), split="train")
    eval_data = load_dataset("json", data_files=str(args.validation), split="train")
    train_data = train_data.map(
        prepare_prompt_completion,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=train_data.column_names,
        desc="Preparing caretaker training examples",
    )
    eval_data = eval_data.map(
        prepare_prompt_completion,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=eval_data.column_names,
        desc="Preparing caretaker validation examples",
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        inference_mode=False,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
    )
    training = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        warmup_steps=1,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        report_to="none",
        dataloader_num_workers=0,
        pad_token=tokenizer.pad_token,
        eos_token=tokenizer.eos_token,
    )
    trainer = SFTTrainer(
        model=model,
        args=training,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=lora,
    )
    print(f"Starting training with {train_count} train and {eval_count} validation examples")
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Phi 4 Mini LoRA adapter saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
