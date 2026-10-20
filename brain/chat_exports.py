import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document


@dataclass
class ParsedMessage:
    sender: str
    text: str
    timestamp: str


# Matches both Android ("15/02/26, 09:14 - Name: msg") and iOS ("[15/02/26, 09:14:00] Name: msg")
_WA_LINE = re.compile(
    r"^(?:\[(?P<bdt>[^\]]+)\]\s*|(?P<adt>\d{1,2}[./]\d{1,2}[./]\d{2,4},?\s+\d{1,2}:\d{2}"
    r"(?::\d{2})?(?:\s*[APap][Mm])?)\s*-\s*)"
    r"(?P<sender>[^:]{1,80}?):\s?(?P<text>.*)$"
)


def parse_whatsapp(path: Path) -> list[ParsedMessage]:
    messages: list[ParsedMessage] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _WA_LINE.match(line.strip())
        if m:
            ts = (m.group("bdt") or m.group("adt") or "").strip()
            messages.append(ParsedMessage(m.group("sender").strip(), m.group("text").strip(), ts))
        elif messages and line.strip():
            messages[-1].text += "\n" + line.strip()  # multiline continuation
    return messages


def _flatten_text(text_field) -> str:
    if isinstance(text_field, str):
        return text_field
    parts = []
    for part in text_field or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            parts.append(str(part.get("text", "")))
    return "".join(parts)


def parse_telegram_json(path: Path) -> list[ParsedMessage]:
    data = json.loads(path.read_text(encoding="utf-8"))
    messages = []
    for m in data.get("messages", []):
        if m.get("type") != "message":
            continue
        text = _flatten_text(m.get("text")).strip()
        if not text:
            continue
        messages.append(ParsedMessage(m.get("from") or m.get("actor") or "unknown", text, m.get("date", "")))
    return messages


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ts_key(ts: str):
    # dd/mm/yy[, HH:MM[:SS]] (WhatsApp dd/mm convention); unknown formats sort last
    m = re.match(
        r"(\d{1,2})[./](\d{1,2})[./](\d{2,4})(?:,\s*(\d{1,2}):(\d{2})(?::(\d{2}))?)?", ts)
    if not m:
        return (1, ts)
    d, mo, y, hh, mm, ss = m.groups()
    year = int(y)
    year += 2000 if year < 100 else 0
    return (0, year, int(mo), int(d), int(hh or 0), int(mm or 0), int(ss or 0))


def detect_and_parse(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            msgs = parse_telegram_json(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            app, chat_name = "telegram", str(data.get("name") or path.stem)
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            return []
    elif suffix == ".txt":
        msgs, app, chat_name = parse_whatsapp(path), "whatsapp", path.stem
    else:
        return []
    if not msgs:
        return []
    body = "\n".join(f"{m.sender}: {m.text}" for m in msgs)
    dates = sorted({m.timestamp for m in msgs if m.timestamp}, key=_ts_key)
    date_range = f"{dates[0]} .. {dates[-1]}" if dates else "unknown"
    return [Document(
        text=body,
        metadata={"category": "chats", "app": app, "chat_name": chat_name,
                  "date_range": date_range,
                  "file_name": path.name, "file_hash": _file_hash(path)},
    )]
