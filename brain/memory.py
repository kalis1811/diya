"""Small, user-controlled long-term memory for facts and corrections."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import Lock
from uuid import uuid4

_WRITE_LOCK = Lock()
_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_STOPWORDS = frozenset(
    "a an and are as at be been but by did do does for from has have how i in is "
    "it its my of on or that the their them this to was we were what when where "
    "which who why with you your".split())
_REMEMBER_RE = re.compile(
    r"^\s*(?:please\s+)?(?:remember(?:\s+that)?|don't\s+forget(?:\s+that)?)\s+(.+?)\s*[.!?]*\s*$",
    re.IGNORECASE)
_PREFERENCE_RE = re.compile(r"\b(i\s+prefer|i\s+like|please\s+(?:answer|explain|write))\b", re.IGNORECASE)
MAX_MEMORIES = 500
MAX_MEMORY_CHARS = 2000


def parse_remember_command(text: str) -> str | None:
    match = _REMEMBER_RE.match(text)
    if not match:
        return None
    memory = match.group(1).strip()
    return memory if memory else None


def inferred_kind(text: str) -> str:
    return "preference" if _PREFERENCE_RE.search(text) else "fact"


class MemoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def get_all(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(f"Could not read personal memory at {self.path}: {e}") from e
        if not isinstance(data, list):
            raise RuntimeError(f"Personal memory at {self.path} is not a list.")
        return [item for item in data if isinstance(item, dict) and
                isinstance(item.get("text"), str) and item.get("kind") in
                {"fact", "preference", "correction"}]

    def add(self, text: str, kind: str = "fact") -> dict:
        text = " ".join(text.split())
        if not text:
            raise ValueError("Memory text must not be blank.")
        if len(text) > MAX_MEMORY_CHARS:
            raise ValueError(f"A memory must be {MAX_MEMORY_CHARS} characters or fewer.")
        if kind not in {"fact", "preference", "correction"}:
            raise ValueError("Memory kind must be fact, preference, or correction.")
        with _WRITE_LOCK:
            records = self.get_all()
            duplicate = next((item for item in records
                              if item["kind"] == kind and
                              item["text"].casefold() == text.casefold()), None)
            if duplicate:
                return duplicate
            if len(records) >= MAX_MEMORIES:
                raise ValueError(f"Memory is full ({MAX_MEMORIES} items). Remove older items first.")
            record = {"text": text, "kind": kind,
                      "id": uuid4().hex,
                      "created_at": datetime.now(timezone.utc).isoformat()}
            records.append(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(self.path.suffix + ".tmp")
            temp.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            temp.replace(self.path)
            return record

    def delete(self, memory_id: str) -> bool:
        with _WRITE_LOCK:
            records = self.get_all()
            remaining = [item for item in records if item.get("id") != memory_id]
            if len(remaining) == len(records):
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(self.path.suffix + ".tmp")
            temp.write_text(json.dumps(remaining, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            temp.replace(self.path)
            return True

    def recall(self, question: str, limit: int = 8) -> list[dict]:
        records = self.get_all()
        terms = {word.casefold() for word in _WORD_RE.findall(question)
                 if len(word) > 2 and word.casefold() not in _STOPWORDS}
        ranked = []
        for position, item in enumerate(records):
            memory_terms = {word.casefold() for word in _WORD_RE.findall(item["text"])
                            if len(word) > 2 and word.casefold() not in _STOPWORDS}
            overlap = len(terms & memory_terms)
            if item["kind"] == "preference":
                # Communication preferences apply broadly across questions.
                score = 1.0 + overlap
            elif overlap:
                score = float(overlap)
            else:
                continue
            ranked.append((score, position, item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [item for _, _, item in ranked[:limit]]
