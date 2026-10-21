# Diya caretaker integration audit

**Date:** 2026-10-21

## Existing architecture

- FastAPI backend: `brain/api.py`; the service exposes health, RAG, memory, conversation, knowledge, voice, and browser-action endpoints.
- RAG: LlamaIndex retrieval with Qdrant plus lexical retrieval in `brain/engine.py`; ingestion and document reading are separate modules.
- Personal memory and conversation history: `brain/memory.py` and `brain/storage.py`, with MongoDB support and local JSON fallback for memories.
- Planning already present: `brain/scheduling.py` has deterministic appointment-planning logic. It is not a general task ledger.
- Existing API middleware protects all routes except `/health` with the configured single-owner Basic login.
- Existing tests cover API, RAG, ingestion, memory behavior indirectly, voice, OCR, and browser actions. The dependency lock is `uv.lock`; no CI workflow is present in the current GitHub tree.

## Duplicate check and gap

The current `origin/main` tree has no generic task model, task state transitions, autonomy levels, or task audit ledger. Existing memory persistence and appointment scheduling have distinct purposes, so they were retained. The caretaker policy is implemented once on `AutonomyLevel` and reused by the ledger and API.

## Implemented update

- Added bounded task records with pending, running, completed, failed, cancelled, and approval-required states.
- Added rule-based autonomy levels (0–3); level 3 consequential actions require explicit approval. API policy uses the configured level as a ceiling so a task cannot grant itself more autonomy.
- Added a local atomic JSON task store at `data/tasks.json`, separate from memory and the RAG index, plus metadata-only audit events and deletion support.
- Added authenticated task CRUD/state/step/audit/authorization endpoints and configuration validation for `BRAIN_AUTONOMY_LEVEL`.
- Task completion requires verification evidence. Step recording is bounded and rejects unplanned or duplicate step IDs.

## Limits and next phase

Task state and audit records are local JSON and process-locked only; run one API process for writes. Objective and scope text persist in plaintext under ignored `data/`; users should not include secrets. Completion evidence is caller-provided and is not independently verified by Diya yet. The new endpoints manage bookkeeping state; they do not invoke tools or execute tasks. There is no scheduler, proactive monitoring, live cancellation, or automated recovery. Next, add a typed allow-listed tool registry and bounded executor only after the app's existing action boundaries are reviewed.
