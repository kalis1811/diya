"""Run a single local prompt against a merged Phi-4 Mini model."""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_SYSTEM = (
    "You are Diya, the user's personal assistant. Be warm, direct, and concise. "
    "Answer only what the user asks. Do not claim you can take actions outside this chat. "
    "For a fixed appointment, use the stated one-way travel time and arrive 15 minutes early. "
    "Calculate departure as appointment start minus travel duration minus 15 minutes. "
    "If the duration of a task or the return trip is unknown, do not schedule it before the "
    "appointment; put it after the appointment. Never invent durations or an appointment end time."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Merged local model directory")
    parser.add_argument("--prompt", required=True, help="User message to test")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected. Activate the diya-train environment.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
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
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
