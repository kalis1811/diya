"""Persistent task state and deterministic caretaker permission policy."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from threading import RLock
from uuid import uuid4


class AutonomyLevel(IntEnum):
    OBSERVE = 0
    RECOMMEND = 1
    ROUTINE = 2
    SUPERVISED = 3

    def permits(self, *, consequential: bool = False, user_approved: bool = False) -> bool:
        """Check an action against a level set by trusted application config."""
        if consequential:
            return self is AutonomyLevel.SUPERVISED and user_approved
        return self >= AutonomyLevel.ROUTINE


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    APPROVAL_REQUIRED = "approval_required"


class InvalidTransition(ValueError):
    """Raised when a task is moved through an unsupported state transition."""


@dataclass
class Task:
    objective: str
    scope: str
    planned_steps: tuple[str, ...]
    autonomy: AutonomyLevel
    max_steps: int = 10
    task_id: str = field(default_factory=lambda: str(uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    completed_steps: list[str] = field(default_factory=list)
    verification_evidence: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: _now())
    updated_at: str = field(default_factory=lambda: _now())
    due_at: str | None = None
    remind_at: str | None = None
    last_nudged_at: str | None = None

    def __post_init__(self) -> None:
        self.autonomy = AutonomyLevel(self.autonomy)
        self.objective = self.objective.strip()
        self.scope = self.scope.strip()
        self.planned_steps = tuple(step.strip() for step in self.planned_steps)
        if not self.objective:
            raise ValueError("objective must not be empty")
        if not self.scope:
            raise ValueError("scope must not be empty")
        if not 1 <= self.max_steps <= 50:
            raise ValueError("max_steps must be between 1 and 50")
        if len(self.planned_steps) > self.max_steps:
            raise ValueError("planned steps exceed max_steps")
        if any(not step for step in self.planned_steps):
            raise ValueError("planned step identifiers must not be empty")
        if len(set(self.planned_steps)) != len(self.planned_steps):
            raise ValueError("planned step identifiers must be unique")
        self.due_at = _normalize_time(self.due_at, "due_at")
        self.remind_at = _normalize_time(self.remind_at, "remind_at")
        self.last_nudged_at = _normalize_time(self.last_nudged_at, "last_nudged_at")

    def transition(
        self,
        target: TaskStatus,
        *,
        user_approved: bool = False,
        verification_evidence: tuple[str, ...] = (),
    ) -> None:
        allowed = {
            TaskStatus.PENDING: {TaskStatus.RUNNING, TaskStatus.APPROVAL_REQUIRED, TaskStatus.CANCELLED},
            TaskStatus.RUNNING: {
                TaskStatus.COMPLETED, TaskStatus.FAILED,
                TaskStatus.CANCELLED, TaskStatus.APPROVAL_REQUIRED,
            },
            TaskStatus.APPROVAL_REQUIRED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
            TaskStatus.COMPLETED: set(),
            TaskStatus.FAILED: set(),
            TaskStatus.CANCELLED: set(),
        }
        if target not in allowed[self.status]:
            raise InvalidTransition(f"cannot transition {self.status.value} to {target.value}")
        if self.status == TaskStatus.APPROVAL_REQUIRED and target == TaskStatus.RUNNING and not user_approved:
            raise PermissionError("explicit user approval is required to resume this task")
        evidence = tuple(item.strip() for item in verification_evidence if item.strip())
        if target == TaskStatus.COMPLETED and not set(self.planned_steps).issubset(self.completed_steps):
            raise ValueError("all planned steps must be recorded before completion")
        if target == TaskStatus.COMPLETED and not evidence:
            raise ValueError("completion requires verification evidence")
        self.status = target
        if target == TaskStatus.COMPLETED:
            self.verification_evidence = evidence
        self.updated_at = _now()

    def record_step(self, step_id: str) -> None:
        if self.status != TaskStatus.RUNNING:
            raise InvalidTransition("steps can only be recorded while a task is running")
        if step_id not in self.planned_steps:
            raise ValueError("step_id is not part of this task plan")
        if step_id in self.completed_steps:
            raise ValueError("step_id was already recorded")
        if len(self.completed_steps) >= self.max_steps:
            raise ValueError("task step limit reached")
        self.completed_steps.append(step_id)
        self.updated_at = _now()


@dataclass(frozen=True)
class AuditEvent:
    task_id: str
    event: str
    status: TaskStatus
    timestamp: str
    step_id: str | None = None


class CaretakerLedger:
    """Atomic JSON task store. Audit events contain metadata, not task text."""

    MAX_TASKS = 500
    MAX_EVENTS = 2000

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = RLock()
        self._tasks: dict[str, Task] = {}
        self._events: list[AuditEvent] = []
        self._load()

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def list_tasks(self) -> tuple[Task, ...]:
        with self._lock:
            return tuple(sorted(self._tasks.values(), key=lambda task: task.created_at, reverse=True))

    def create_task(self, task: Task) -> Task:
        with self._lock:
            if task.task_id in self._tasks:
                raise ValueError("task_id already exists")
            if len(self._tasks) >= self.MAX_TASKS:
                raise ValueError("task store is full; delete completed tasks before adding more")
            events_before = self._events.copy()
            self._tasks[task.task_id] = task
            self._record(task, "task_created")
            self._save_or_restore(task.task_id, None, events_before)
            return task

    def get_task(self, task_id: str) -> Task:
        with self._lock:
            try:
                return self._tasks[task_id]
            except KeyError as exc:
                raise KeyError("task not found") from exc

    def delete_task(self, task_id: str) -> bool:
        with self._lock:
            if task_id not in self._tasks:
                return False
            task_before = _task_from_dict(_task_to_dict(self._tasks[task_id]))
            events_before = self._events.copy()
            del self._tasks[task_id]
            self._events = [event for event in self._events if event.task_id != task_id]
            self._save_or_restore(task_id, task_before, events_before)
            return True

    def authorize_action(
        self,
        task_id: str,
        *,
        consequential: bool = False,
        user_approved: bool = False,
        policy_level: AutonomyLevel | None = None,
    ) -> bool:
        with self._lock:
            task = self.get_task(task_id)
            task_before = _task_from_dict(_task_to_dict(task))
            events_before = self._events.copy()
            effective_level = min(task.autonomy, policy_level) if policy_level is not None else task.autonomy
            permitted = effective_level.permits(
                consequential=consequential,
                user_approved=user_approved,
            )
            self._record(task, "action_authorized" if permitted else "action_denied")
            self._save_or_restore(task_id, task_before, events_before)
            return permitted

    def transition(self, task_id: str, target: TaskStatus, **kwargs) -> Task:
        with self._lock:
            task = self.get_task(task_id)
            task_before = _task_from_dict(_task_to_dict(task))
            events_before = self._events.copy()
            task.transition(target, **kwargs)
            self._record(task, f"status:{target.value}")
            self._save_or_restore(task_id, task_before, events_before)
            return task

    def record_step(self, task_id: str, step_id: str) -> None:
        with self._lock:
            task = self.get_task(task_id)
            task_before = _task_from_dict(_task_to_dict(task))
            events_before = self._events.copy()
            task.record_step(step_id)
            self._record(task, "step_recorded", step_id)
            self._save_or_restore(task_id, task_before, events_before)

    def claim_due_nudges(self, now: datetime | None = None) -> list[dict[str, str]]:
        """Return and persist due caretaker messages once, in explainable priority order."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must include a timezone")
        now = now.astimezone(timezone.utc)
        with self._lock:
            candidates = []
            for task in self._tasks.values():
                if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                    continue
                due = _as_datetime(task.due_at)
                reminder = _as_datetime(task.remind_at)
                updated = _as_datetime(task.updated_at)
                last_nudged = _as_datetime(task.last_nudged_at)
                kind = None
                trigger = None
                priority = 99
                if task.status is TaskStatus.APPROVAL_REQUIRED and (
                    last_nudged is None or (updated and last_nudged < updated)
                ):
                    kind, trigger, priority = "approval", updated or now, 0
                elif reminder and reminder <= now and (
                    last_nudged is None or last_nudged < reminder
                ):
                    kind, trigger, priority = "reminder", reminder, 1
                elif due and due <= now and (
                    last_nudged is None or last_nudged < due
                ):
                    kind, trigger, priority = "overdue", due, 2
                if kind is None:
                    continue
                candidates.append((priority, trigger, task.created_at, task, kind))

            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            if not candidates:
                return []
            events_before = self._events.copy()
            originals = {
                item[3].task_id: (item[3].last_nudged_at, item[3].updated_at)
                for item in candidates
            }
            nudges = []
            for _priority, _trigger, _created, task, kind in candidates:
                task.last_nudged_at = now.isoformat()
                task.updated_at = now.isoformat()
                self._record(task, f"nudge_sent:{kind}")
                if kind == "approval":
                    message = f"I’m waiting for your approval before continuing: {task.objective}."
                    reason = "This task requires your approval before it can continue."
                elif kind == "reminder":
                    message = f"A reminder you set: {task.objective}."
                    reason = "Its scheduled reminder time has arrived."
                else:
                    message = f"This task is past its due time: {task.objective}."
                    reason = "The task's due time has passed."
                nudges.append({
                    "task_id": task.task_id,
                    "kind": kind,
                    "objective": task.objective,
                    "message": message,
                    "reason": reason,
                    "trigger_at": _trigger.isoformat(),
                })
            try:
                self._save()
            except OSError:
                for task in self._tasks.values():
                    if task.task_id in originals:
                        task.last_nudged_at, task.updated_at = originals[task.task_id]
                self._events = events_before
                raise
            return nudges

    def _save_or_restore(
        self,
        task_id: str,
        task_before: Task | None,
        events_before: list[AuditEvent],
    ) -> None:
        try:
            self._save()
        except OSError:
            if task_before is None:
                self._tasks.pop(task_id, None)
            else:
                self._tasks[task_id] = task_before
            self._events = events_before
            raise

    def _record(self, task: Task, event: str, step_id: str | None = None) -> None:
        self._events.append(AuditEvent(task.task_id, event, task.status, _now(), step_id))
        self._events = self._events[-self.MAX_EVENTS:]

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                raise ValueError("unsupported data version")
            tasks = [_task_from_dict(item) for item in data["tasks"]]
            events = [AuditEvent(
                task_id=item["task_id"], event=item["event"],
                status=TaskStatus(item["status"]), timestamp=item["timestamp"],
                step_id=item.get("step_id"),
            ) for item in data["events"]]
            self._tasks = {task.task_id: task for task in tasks}
            if len(self._tasks) != len(tasks):
                raise ValueError("duplicate task identifiers")
            self._events = events[-self.MAX_EVENTS:]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not load caretaker task state from {self.path}") from exc

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "tasks": [_task_to_dict(task) for task in self._tasks.values()],
            "events": [
                {**asdict(event), "status": event.status.value}
                for event in self._events
            ],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_to_dict(task: Task) -> dict:
    result = asdict(task)
    result["autonomy"] = int(task.autonomy)
    result["status"] = task.status.value
    result["planned_steps"] = list(task.planned_steps)
    result["verification_evidence"] = list(task.verification_evidence)
    return result


def _task_from_dict(item: dict) -> Task:
    return Task(
        objective=item["objective"], scope=item["scope"],
        planned_steps=tuple(item["planned_steps"]),
        autonomy=AutonomyLevel(item["autonomy"]), max_steps=item["max_steps"],
        task_id=item["task_id"], status=TaskStatus(item["status"]),
        completed_steps=list(item["completed_steps"]),
        verification_evidence=tuple(item["verification_evidence"]),
        created_at=item["created_at"], updated_at=item["updated_at"],
        due_at=item.get("due_at"), remind_at=item.get("remind_at"),
        last_nudged_at=item.get("last_nudged_at"),
    )


def _as_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _normalize_time(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()
