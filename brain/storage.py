"""Mongo-backed conversation and memory storage with a local memory fallback."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from uuid import uuid4

from brain.memory import MemoryStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BrainStorage:
    def __init__(self, uri: str, database: str, memory_path: Path):
        self._uri = uri
        self._database = database
        self._fallback = MemoryStore(memory_path)
        self._client = None
        self._db = None
        self.status = "local file fallback (MongoDB URI not configured)"
        if not uri:
            return
        self._connect()

    def _connect(self) -> None:
        if not self._uri:
            return
        try:
            from pymongo import MongoClient
            client = MongoClient(self._uri, serverSelectionTimeoutMS=1200,
                                connectTimeoutMS=1200)
            client.admin.command("ping")
            self._client = client
            self._db = client[self._database]
            self._db.memories.create_index([("kind", 1), ("text_key", 1)])
            self._db.conversations.create_index([("updated_at", -1)])
            self._db.messages.create_index([("conversation_id", 1), ("created_at", 1)])
            self._migrate_local_memories()
            self.status = f"MongoDB connected ({self._database})"
        except Exception as exc:
            self._db = None
            self._client = None
            self.status = f"MongoDB unavailable; using local memory fallback ({exc.__class__.__name__})"

    def refresh(self) -> None:
        """Retry an unavailable local MongoDB after it has been started."""
        if self._db is None and self._uri:
            self._connect()
        elif self._db is not None:
            try:
                self._client.admin.command("ping")
            except Exception as exc:
                self._db = None
                self._client = None
                self.status = f"MongoDB unavailable; using local memory fallback ({exc.__class__.__name__})"

    @property
    def mongo_ready(self) -> bool:
        return self._db is not None

    def _migrate_local_memories(self) -> None:
        """Copy existing file memories into MongoDB without deleting the source."""
        for item in self._fallback.get_all():
            key = item["text"].casefold()
            self._db.memories.update_one(
                {"kind": item["kind"], "text_key": key},
                {"$setOnInsert": {"_id": item["id"], "text": item["text"],
                                   "text_key": key, "kind": item["kind"],
                                   "created_at": item["created_at"]}}, upsert=True)

    @property
    def memories(self):
        return self

    def get_all(self) -> list[dict]:
        if self._db is None:
            return self._fallback.get_all()
        return [{"id": row["_id"], "text": row["text"], "kind": row["kind"],
                 "created_at": row["created_at"]}
                for row in self._db.memories.find({}, {"text": 1, "kind": 1,
                                                       "created_at": 1}).sort("created_at", 1)]

    def add(self, text: str, kind: str = "fact") -> dict:
        if self._db is None:
            return self._fallback.add(text, kind)
        text = " ".join(text.split())
        if not text:
            raise ValueError("Memory text must not be blank.")
        if len(text) > 2000:
            raise ValueError("A memory must be 2000 characters or fewer.")
        if kind not in {"fact", "preference", "correction"}:
            raise ValueError("Memory kind must be fact, preference, or correction.")
        key = text.casefold()
        duplicate = self._db.memories.find_one({"kind": kind, "text_key": key})
        if duplicate:
            return {"id": duplicate["_id"], "text": duplicate["text"],
                    "kind": duplicate["kind"], "created_at": duplicate["created_at"]}
        if self._db.memories.count_documents({}) >= 500:
            raise ValueError("Memory is full (500 items). Remove older items first.")
        item = {"_id": uuid4().hex, "text": text, "text_key": key,
                "kind": kind, "created_at": _now()}
        self._db.memories.insert_one(item)
        return {"id": item["_id"], "text": text, "kind": kind,
                "created_at": item["created_at"]}

    def delete(self, memory_id: str) -> bool:
        if self._db is None:
            return self._fallback.delete(memory_id)
        return self._db.memories.delete_one({"_id": memory_id}).deleted_count > 0

    def recall(self, question: str, limit: int = 8) -> list[dict]:
        if self._db is None:
            return self._fallback.recall(question, limit)
        import re
        stop = {"the", "and", "for", "are", "but", "not", "you", "your", "with", "from", "that", "this", "what", "where", "when", "how", "can", "does", "have", "about"}
        terms = {w.casefold() for w in re.findall(r"\w+", question)
                 if len(w) > 2 and w.casefold() not in stop}
        ranked = []
        for position, item in enumerate(self.get_all()):
            words = {w.casefold() for w in re.findall(r"\w+", item["text"])
                     if len(w) > 2 and w.casefold() not in stop}
            overlap = len(terms & words)
            score = 1.0 + overlap if item["kind"] == "preference" else float(overlap)
            if score:
                ranked.append((score, position, item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [item for _, _, item in ranked[:limit]]

    def create_conversation(self, title: str = "New conversation") -> dict:
        if self._db is None:
            raise RuntimeError(self.status)
        now = _now()
        item = {"_id": uuid4().hex, "title": title, "created_at": now,
                "updated_at": now, "preview": "", "message_count": 0}
        self._db.conversations.insert_one(item)
        return self._conversation_dict(item)

    @staticmethod
    def _conversation_dict(item: dict) -> dict:
        return {"id": item["_id"], "title": item["title"],
                "created_at": item["created_at"], "updated_at": item["updated_at"],
                "preview": item.get("preview", ""),
                "message_count": item.get("message_count", 0)}

    def list_conversations(self, limit: int = 40) -> list[dict]:
        if self._db is None:
            return []
        return [self._conversation_dict(row) for row in
                self._db.conversations.find().sort("updated_at", -1).limit(limit)]

    def get_conversation(self, conversation_id: str) -> dict | None:
        if self._db is None:
            return None
        row = self._db.conversations.find_one({"_id": conversation_id})
        return self._conversation_dict(row) if row else None

    def append_message(self, conversation_id: str, role: str, content: str,
                       sources: list[dict] | None = None) -> None:
        if self._db is None:
            return
        if role not in {"user", "assistant"}:
            raise ValueError("Invalid conversation role")
        now = _now()
        self._db.messages.insert_one({"_id": uuid4().hex,
                                      "conversation_id": conversation_id,
                                      "role": role, "content": content,
                                      "sources": sources or [], "created_at": now})
        title = content.strip().replace("\n", " ")[:64]
        self._db.conversations.update_one(
            {"_id": conversation_id},
            {"$set": {"updated_at": now, "preview": title},
             "$inc": {"message_count": 1}}, upsert=True)
        if role == "user":
            self._db.conversations.update_one(
                {"_id": conversation_id, "title": "New conversation"},
                {"$set": {"title": title or "New conversation"}})

    def conversation_messages(self, conversation_id: str) -> list[dict]:
        if self._db is None or self.get_conversation(conversation_id) is None:
            return []
        return [{"id": row["_id"], "role": row["role"], "content": row["content"],
                 "sources": row.get("sources", []), "created_at": row["created_at"]}
                for row in self._db.messages.find({"conversation_id": conversation_id})
                .sort("created_at", 1)]

    def delete_conversation(self, conversation_id: str) -> bool:
        if self._db is None:
            return False
        result = self._db.conversations.delete_one({"_id": conversation_id})
        if result.deleted_count:
            self._db.messages.delete_many({"conversation_id": conversation_id})
        return result.deleted_count > 0


_STORAGE_LOCK = Lock()


@lru_cache(maxsize=4)
def _cached_storage(uri: str, database: str, memory_path: str) -> BrainStorage:
    return BrainStorage(uri, database, Path(memory_path))


def get_storage(settings) -> BrainStorage:
    with _STORAGE_LOCK:
        return _cached_storage(getattr(settings, "mongodb_uri", ""),
                               getattr(settings, "mongodb_database", "diya"),
                               str(settings.memory_path))
