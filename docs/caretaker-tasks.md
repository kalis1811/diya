# Caretaker task state

Diya's existing FastAPI service exposes a small authenticated task ledger. It
stores task execution state separately from RAG documents and long-term memory
in `data/tasks.json` (or the sibling of the configured raw data directory).
Writes use an atomic temporary-file replacement. Audit events store task IDs,
state changes, step IDs, and timestamps; they do not copy objectives or
verification text.

Set `BRAIN_AUTONOMY_LEVEL` in the local `.env` to an integer from `0` to `3`.
The default is `0` (observe). The LLM cannot change this setting. A task records
the configured level at creation, and the current configured level caps it when
an action permission is checked. Consequential actions require level `3` and
explicit approval. Diya does not execute task steps through this API yet; step
records and status transitions are bookkeeping only.

All endpoints below use the existing single-owner Basic authentication. The
`/health` endpoint remains public as before.

| Endpoint | Purpose |
| --- | --- |
| `GET /tasks` | List tasks |
| `POST /tasks` | Create a task with objective, scope, planned steps, and a bounded step limit |
| `GET /tasks/{task_id}` | Read task state and verification evidence |
| `POST /tasks/{task_id}/transition` | Move task state; completion requires verification evidence |
| `POST /tasks/{task_id}/steps` | Record a planned step once while running |
| `POST /tasks/{task_id}/authorize` | Check a routine or consequential action against policy and record the decision |
| `GET /tasks/{task_id}/events` | Read that task's metadata-only audit events |
| `DELETE /tasks/{task_id}` | Delete a task and its audit events |

Task objectives and scope are stored locally in plaintext in the ignored `data`
directory so unfinished tasks can survive a restart. Do not put credentials or
other secrets in task text. The current JSON store has process-local locking;
run a single API process when writing task state. Scheduling, tool execution,
timeouts, cancellation of live work, automated recovery, and proactive checks
are not implemented by this task ledger. Completion evidence is supplied by the
authenticated caller; Diya does not independently validate it until a verified
tool execution layer is added.

## Reminder and decision rules

The Reminders view lets the owner schedule a one-time prompt. Diya's visible app
polls `GET /caretaker/nudges` every 30 seconds while the page is open and visible.
The API claims due prompts in the local task ledger, so a reminder is delivered
once even across a restart. Keep the app open to see prompts near their
scheduled time; there is no operating-system notification service or background
delivery when the browser is closed.

The decision order is deterministic and explainable:

1. A task in `approval_required` is surfaced first, with the reason that Diya is
   waiting for approval.
2. A user-scheduled `remind_at` time creates a reminder.
3. A non-terminal task past `due_at` gets an overdue prompt.

Each prompt includes its trigger and a short reason. The LLM does not invent
reminders or decide to interrupt the owner, and these prompts never execute
task steps. Times from the web form are converted to UTC before storage. API
clients must send ISO-8601 timestamps with a timezone offset.
