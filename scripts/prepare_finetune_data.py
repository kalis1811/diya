"""Prepare a local, user-labelled chat dataset for later personal-style tuning.

This script never uploads data or starts model training. It writes JSONL files
under data/finetune by default; the data directory is ignored by Git.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from brain.chat_exports import parse_telegram_json, parse_whatsapp


SUPPORTED = {".txt", ".json"}
MAX_CHAT_TEXT_CHARS = 500
FORWARDED_RE = re.compile(r"\[\s*forwarded\s*\]", re.IGNORECASE)
VCARD_RE = re.compile(r"\bBEGIN:VCARD\b|\bEND:VCARD\b", re.IGNORECASE)
TIMESTAMP_ARTIFACT_RE = re.compile(
    r"\s*\[\s*\d{1,2}[./]\d{1,2}[./]\d{2,4},?\s+\d{1,2}:\d{2}"
    r"(?::\d{2})?\s*(?:AM|PM)?\s*\]\s*-\s*",
    re.IGNORECASE,
)
SYSTEM_NOTICE_RE = re.compile(
    r"messages and calls are end-to-end encrypted|"
    r"you received a message on your phone, but your version of whatsapp|"
    r"this message was deleted|you deleted this message|\[system notification\]",
    re.IGNORECASE,
)
ATTACHMENT_NAME_RE = re.compile(
    r"[\w .()\-]+\.(?:zip|pdf|docx?|xlsx?|pptx?|jpe?g|png|mp4|mp3|txt)",
    re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(
    r"<\s*(?:image|video|audio|media|sticker|document|album|gif|call|voice|contact)"
    r"(?:\s+(?:message|omitted|attached))?\s*>"
    r"|\[\s*(?:image|video|audio|media|sticker|document|album|gif|call|voice|contact)"
    r"(?:\s+(?:message|omitted|attached))?\s*\]"
    r"|\b(?:image|video|audio|media|sticker|document)\s+omitted\b"
    r"|<\s*attached:\s*[^>]+>",
    re.IGNORECASE,
)
PRIVATE_DETAIL_RE = re.compile(
    r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"
    r"|(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)"
    r"|(?<!\d)-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}(?!\d)",
    re.IGNORECASE,
)


def read_export(path: Path):
    if path.suffix.lower() == ".txt":
        return parse_whatsapp(path)
    if path.suffix.lower() == ".json":
        return parse_telegram_json(path)
    return []


def clean_chat_text(value: str) -> str | None:
    """Drop non-conversational/private content and strip export artifacts."""
    text = TIMESTAMP_ARTIFACT_RE.sub(" ", value)
    text = " ".join(text.split())
    if (not text or FORWARDED_RE.search(text) or VCARD_RE.search(text)
            or PRIVATE_DETAIL_RE.search(text) or SYSTEM_NOTICE_RE.search(text)
            or ATTACHMENT_NAME_RE.fullmatch(text)
            or len(text) > MAX_CHAT_TEXT_CHARS):
        return None
    text = " ".join(PLACEHOLDER_RE.sub(" ", text).split())
    if not text or re.fullmatch(r"[\W_]*", text):
        return None
    return text


def make_examples(paths: list[Path], user_names: set[str], context_messages: int):
    by_export = {}
    skipped = 0
    filtered_messages = 0
    senders = set()
    for path in paths:
        try:
            messages = read_export(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
                AttributeError, TypeError):
            skipped += 1
            continue
        examples = []
        history: list[dict[str, str]] = []
        for message in messages:
            senders.add(message.sender.strip())
            text = clean_chat_text(message.text)
            if text is None:
                filtered_messages += 1
                continue
            is_user = message.sender.strip().casefold() in user_names
            if is_user:
                # Train only direct replies to a newly incoming message. This
                # prevents self-followups from becoming misleading targets.
                if history and history[-1]["role"] == "user":
                    context = history[-context_messages:]
                    turns = [{"role": item["role"], "content": item["content"]}
                             for item in context]
                    turns.append({"role": "assistant", "content": text})
                    examples.append({"messages": turns})
                history.append({"role": "assistant", "content": text})
            else:
                history.append({"role": "user", "content": text})
        if examples:
            by_export[str(path.resolve())] = examples
    return by_export, skipped, senders, filtered_messages


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/chats"),
                        help="folder containing WhatsApp TXT or Telegram JSON exports")
    parser.add_argument("--output", type=Path, default=Path("data/finetune"),
                        help="local output folder (default: ignored data/finetune)")
    parser.add_argument("--user-name", action="append", required=True,
                        help="your exact sender name in the export; repeat for name variants")
    parser.add_argument("--context-messages", type=int, default=6,
                        help="maximum preceding messages per example (default: 6)")
    parser.add_argument("--validation-fraction", type=float, default=0.1,
                        help="fraction held out for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.context_messages < 1:
        parser.error("--context-messages must be at least 1")
    if not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be at least 0 and less than 1")
    if not args.input.is_dir():
        parser.error(f"input folder does not exist: {args.input}")

    user_names = {name.strip().casefold() for name in args.user_name if name.strip()}
    if not user_names:
        parser.error("at least one non-empty --user-name is required")
    paths = sorted(path for path in args.input.rglob("*")
                   if path.is_file() and path.suffix.lower() in SUPPORTED)
    if not paths:
        parser.error(f"No .txt or .json chat exports found in {args.input}. "
                     "Add a WhatsApp TXT or Telegram JSON export, then retry.")
    by_export, skipped, senders, filtered_messages = make_examples(
        paths, user_names, args.context_messages
    )
    if not senders:
        parser.error("No messages could be read from the supported exports. "
                     "Check the export format and try again.")
    matched_senders = {sender.casefold() for sender in senders} & user_names
    if not matched_senders:
        choices = ", ".join(sorted(senders)[:12])
        parser.error(f"None of the --user-name values matched a sender in the exports. "
                     f"Found: {choices}. Pass your exact sender name with --user-name.")
    if not by_export:
        parser.error("Your sender name matched, but no messages had earlier conversation "
                     "context to create examples from.")
    all_examples = [row for rows in by_export.values() for row in rows]
    exports = list(by_export)
    random.Random(args.seed).shuffle(exports)
    validation_count = int(len(exports) * args.validation_fraction)
    if args.validation_fraction and len(exports) >= 2:
        validation_count = max(1, validation_count)
    validation_exports = set(exports[:validation_count])
    training = [row for name, rows in by_export.items()
                if name not in validation_exports for row in rows]
    validation = [row for name, rows in by_export.items()
                  if name in validation_exports for row in rows]
    write_jsonl(args.output / "all.jsonl", all_examples)
    write_jsonl(args.output / "train.jsonl", training)
    write_jsonl(args.output / "validation.jsonl", validation)
    summary = ("# Personal-style dataset preparation\n\n"
               f"- Supported exports found: {len(paths)}\n"
               f"- Exports skipped due to read/parse errors: {skipped}\n"
               f"- Non-conversational, media, private-detail, or long messages filtered: {filtered_messages}\n"
               f"- Exports with usable examples: {len(by_export)}\n"
               f"- Total examples: {len(all_examples)}\n"
               f"- Training examples: {len(training)}\n"
               f"- Validation examples: {len(validation)}\n"
               f"- Context messages per example: at most {args.context_messages}\n\n"
               "Each target is a reply immediately following an incoming message.\n\n"
               "Review the JSONL locally before training. Check sender labels, "
               "privacy-sensitive content, duplicates, and whether each target "
               "is actually your message. No model was trained by this command.\n")
    (args.output / "stats.md").write_text(summary, encoding="utf-8")
    print(f"Prepared {len(all_examples)} total examples: {len(training)} training and "
          f"{len(validation)} validation.")
    print(f"Review local outputs in: {args.output.resolve()}")


if __name__ == "__main__":
    main()
