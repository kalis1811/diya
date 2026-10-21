import json
import re
import base64
import binascii
import secrets
from threading import Lock
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, field_validator

from brain.browser_actions import run_browser_command
from brain.config import get_settings
from brain.engine import EmptyIndexError, should_use_deep_thinking
from brain.document_reading import safe_filename
from brain.ollama_client import OllamaDown, OllamaError
from brain.document_reading import MAX_UPLOAD_BYTES, extract_and_summarize
from brain.ollama_client import OllamaClient
from brain import persona
from brain.memory import inferred_kind, parse_remember_command
from brain.voice import PiperError, VoskError, resolve_stt, resolve_tts
from brain.storage import get_storage
from brain.scheduling import appointment_schedule
from brain.caretaker import AutonomyLevel, CaretakerLedger, InvalidTransition, Task, TaskStatus


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    deep_think: bool = False
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must contain non-whitespace text")
        return value


class MemoryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    kind: str = Field(default="fact", pattern="^(fact|preference|correction)$")


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1, max_length=800)
    correction: str = Field(min_length=1, max_length=1000)


class TaskCreateRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=1000)
    scope: str = Field(min_length=1, max_length=1000)
    planned_steps: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(
        default_factory=list, max_length=50)
    max_steps: int = Field(default=10, ge=1, le=50)

    @field_validator("objective", "scope")
    @classmethod
    def fields_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must contain non-whitespace text")
        return value


class TaskTransitionRequest(BaseModel):
    status: TaskStatus
    user_approved: bool = False
    verification_evidence: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list, max_length=20)


class TaskStepRequest(BaseModel):
    step_id: str = Field(min_length=1, max_length=100)


class TaskAuthorizationRequest(BaseModel):
    consequential: bool = False
    user_approved: bool = False


class TextToSpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must contain non-whitespace characters")
        return value


MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_HISTORY_TURNS = 8
MAX_HISTORY_MESSAGE_CHARS = 2000


def _recent_history(storage, conversation_id: str | None) -> list[dict]:
    """Return a small, role-filtered window of prior turns for model context."""
    if not conversation_id or not storage.mongo_ready:
        return []
    if storage.get_conversation(conversation_id) is None:
        raise HTTPException(404, "Conversation not found. Start a new chat and try again.")
    messages = storage.conversation_messages(conversation_id)[-MAX_HISTORY_TURNS:]
    return [{"role": item["role"],
             "content": item["content"][-MAX_HISTORY_MESSAGE_CHARS:]}
            for item in messages if item["role"] in {"user", "assistant"}]


def create_app(engine=None, settings=None, browser_opener=None, tts=None, stt=None,
               caretaker=None) -> FastAPI:
    settings = settings or get_settings()
    tts_status = "ready" if tts is not None else None
    if tts is None:
        tts, tts_status = resolve_tts(settings)
    stt_status = "ready" if stt is not None else None
    if stt is None:
        stt, stt_status = resolve_stt(settings)
    engine_lock = Lock()
    storage = get_storage(settings)
    memory_store = storage.memories
    caretaker = caretaker or CaretakerLedger(settings.task_path)

    def current_engine():
        nonlocal engine
        if engine is not None:
            return engine
        with engine_lock:
            if engine is None:
                from brain.engine import load_engine
                try:
                    engine = load_engine(settings)
                except EmptyIndexError:
                    return None
        return engine

    def empty_index_detail() -> str:
        return f"No index at {settings.index_path}. Run: uv run python -m brain.ingest"

    def refresh_knowledge_index():
        """Rebuild the changed portions of the local index while queries are paused."""
        nonlocal engine
        from brain.ingest import run_ingest

        active = engine
        if active is not None and getattr(active, "_qdrant", None) is not None:
            active._qdrant.close()
        engine = None
        try:
            stats = run_ingest(settings)
            try:
                from brain.engine import load_engine
                engine = load_engine(settings)
            except EmptyIndexError:
                engine = None
            return stats
        except Exception:
            # The ingester preserves the previous indexed version if a changed
            # document cannot be embedded. Reopen that index before returning.
            from brain.engine import load_engine
            try:
                engine = load_engine(settings)
            except EmptyIndexError:
                engine = None
            raise

    app = FastAPI(title="brain")

    @app.middleware("http")
    async def require_owner_admin(request: Request, call_next):
        # Keep liveness checks available to the launcher; protect the UI,
        # static assets, and every data-changing or conversational endpoint.
        if request.url.path == "/health":
            return await call_next(request)
        if len(settings.admin_password) < 10:
            return Response(
                "Diya admin login is not configured. Set a 10-character BRAIN_ADMIN_PASSWORD in .env.",
                status_code=503,
                media_type="text/plain",
            )
        header = request.headers.get("authorization", "")
        supplied_user = supplied_password = ""
        scheme, _, encoded = header.partition(" ")
        if scheme.lower() == "basic" and encoded:
            try:
                decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
                supplied_user, separator, supplied_password = decoded.partition(":")
                if not separator:
                    supplied_user = supplied_password = ""
            except (binascii.Error, UnicodeDecodeError):
                supplied_user = supplied_password = ""
        user_ok = secrets.compare_digest(
            supplied_user.encode("utf-8"), settings.admin_username.encode("utf-8"))
        password_ok = secrets.compare_digest(
            supplied_password.encode("utf-8"), settings.admin_password.encode("utf-8"))
        if not (user_ok and password_ok):
            return Response(
                "Admin login required.",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Diya Admin", charset="UTF-8"'},
                media_type="text/plain",
            )
        return await call_next(request)

    @app.get("/")
    def chat_page():
        return FileResponse(
            Path(__file__).parent / "static" / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health")
    def health():
        storage.refresh()
        active_engine = current_engine()
        if active_engine is None:
            return {"status": "needs_ingest", "model": settings.llm_model,
                    "embed_model": settings.embed_model, "indexed_chunks": 0,
                    "voice": {"tts": tts_status, "stt": stt_status}, "storage": storage.status,
                    "admin_configured": len(settings.admin_password) >= 10}
        return {"status": "ok", "model": settings.llm_model,
                "embed_model": settings.embed_model,
                "indexed_chunks": active_engine.chunk_count(),
                "voice": {"tts": tts_status, "stt": stt_status}, "storage": storage.status,
                "admin_configured": len(settings.admin_password) >= 10}

    @app.get("/about")
    def about():
        return {"name": persona.NAME, "model": settings.llm_model,
                "description": persona.self_description(settings.llm_model),
                "memory_items": len(memory_store.get_all()),
                "thinking": "off by default; enable deep_think for a request to opt in"}

    @app.get("/admin/overview")
    def admin_overview():
        storage.refresh()
        active_engine = current_engine()
        conversations = storage.list_conversations()
        return {
            "owner": settings.admin_username,
            "model": settings.llm_model,
            "indexed_chunks": active_engine.chunk_count() if active_engine else 0,
            "memories": len(memory_store.get_all()),
            "conversations": len(conversations),
            "storage": storage.status,
            "voice_output": tts_status or "unavailable",
            "voice_input": stt_status or "unavailable",
        }

    @app.get("/knowledge")
    def list_knowledge():
        folder = settings.raw_dirs["docs"]
        files = []
        if folder.exists():
            for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
                if path.is_file() and not path.name.startswith("."):
                    files.append({"name": path.name, "size": path.stat().st_size})
        active_engine = current_engine()
        with engine_lock:
            active_engine = engine or active_engine
            chunks = active_engine.chunk_count() if active_engine else 0
        return {"files": files,
                "indexed_chunks": chunks}

    @app.post("/knowledge")
    async def add_knowledge(file: UploadFile = File(...)):
        """Save a local source document and add its chunks to the RAG index."""
        nonlocal engine
        filename = file.filename or ""
        try:
            name = safe_filename(filename)
            if Path(name).suffix.lower() not in {
                ".pdf", ".txt", ".md", ".csv", ".docx", ".html", ".htm", ".json"
            }:
                raise ValueError("Supported knowledge files: PDF, TXT, Markdown, CSV, DOCX, HTML, and JSON.")
            data = await file.read(MAX_UPLOAD_BYTES + 1)
            if not data:
                raise ValueError("The selected file is empty.")
            if len(data) > MAX_UPLOAD_BYTES:
                raise ValueError(f"File is too large. The limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        finally:
            await file.close()

        folder = settings.raw_dirs["docs"]
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / name
        if target.is_symlink():
            raise HTTPException(422, "A knowledge file with that name cannot be replaced safely.")
        previous = target.read_bytes() if target.exists() else None
        try:
            with engine_lock:
                target.write_bytes(data)
                stats = refresh_knowledge_index()
                failed = any(f"skipped {name}:" in warning or f"OCR failed for {name}:" in warning
                             for warning in stats.warnings)
                if failed or (stats.files_new == 0 and stats.files_skipped == 0):
                    detail = next((warning for warning in stats.warnings if name in warning),
                                  f"No readable text was found in {name}.")
                    raise ValueError(detail)
        except Exception as e:
            with engine_lock:
                if previous is None:
                    target.unlink(missing_ok=True)
                else:
                    target.write_bytes(previous)
                # Restore the prior source/index pairing if parsing partially succeeded.
                try:
                    refresh_knowledge_index()
                except Exception:
                    if engine is None:
                        from brain.engine import load_engine
                        try:
                            engine = load_engine(settings)
                        except EmptyIndexError:
                            engine = None
            if isinstance(e, HTTPException):
                raise
            if isinstance(e, ValueError):
                raise HTTPException(422, str(e)) from e
            raise HTTPException(503, f"Could not add {name} to the knowledge base: {e}") from e
        return {"message": f"Added {name} to the knowledge base.", "file": name,
                "files_new": stats.files_new, "chunks_indexed": stats.chunks_indexed,
                "warnings": stats.warnings}

    @app.delete("/knowledge/{filename}")
    def delete_knowledge(filename: str):
        nonlocal engine
        try:
            name = safe_filename(filename)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        folder = settings.raw_dirs["docs"]
        target = folder / name
        if target.is_symlink() or not target.is_file() or target.parent.resolve() != folder.resolve():
            raise HTTPException(404, "Knowledge file not found.")
        previous = target.read_bytes()
        try:
            with engine_lock:
                target.unlink()
                stats = refresh_knowledge_index()
        except Exception as e:
            with engine_lock:
                target.write_bytes(previous)
                try:
                    refresh_knowledge_index()
                except Exception:
                    if engine is None:
                        from brain.engine import load_engine
                        try:
                            engine = load_engine(settings)
                        except EmptyIndexError:
                            engine = None
            raise HTTPException(503, f"Could not remove {name} from the knowledge base: {e}") from e
        return {"message": f"Removed {name} from the knowledge base.",
                "chunks_indexed": stats.chunks_indexed, "warnings": stats.warnings}

    @app.get("/conversations")
    def list_conversations():
        return {"conversations": storage.list_conversations(),
                "storage": storage.status}

    @app.post("/conversations")
    def create_conversation():
        if not storage.mongo_ready:
            raise HTTPException(503, f"Conversation storage is unavailable: {storage.status}")
        return storage.create_conversation()

    @app.get("/conversations/{conversation_id}")
    def get_conversation(conversation_id: str):
        conversation = storage.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(404, "Conversation not found.")
        return {**conversation, "messages": storage.conversation_messages(conversation_id)}

    @app.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str):
        if not storage.delete_conversation(conversation_id):
            raise HTTPException(404, "Conversation not found.")
        return {"message": "Conversation deleted."}

    @app.get("/memory")
    def list_memories():
        return {"memories": memory_store.get_all()}

    @app.post("/memory")
    def remember(req: MemoryRequest):
        try:
            item = memory_store.add(req.text, req.kind)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"message": "Saved. I can use this in future answers.", "memory": item}

    @app.delete("/memory/{memory_id}")
    def forget(memory_id: str):
        if not memory_store.delete(memory_id):
            raise HTTPException(404, "Memory not found.")
        return {"message": "Memory deleted."}

    @app.get("/tasks")
    def list_tasks():
        return {"tasks": jsonable_encoder(caretaker.list_tasks())}

    @app.post("/tasks", status_code=201)
    def create_task(req: TaskCreateRequest):
        try:
            task = caretaker.create_task(Task(
                objective=req.objective,
                scope=req.scope,
                planned_steps=tuple(req.planned_steps),
                max_steps=req.max_steps,
                autonomy=AutonomyLevel(settings.autonomy_level),
            ))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return jsonable_encoder(task)

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str):
        try:
            return jsonable_encoder(caretaker.get_task(task_id))
        except KeyError as exc:
            raise HTTPException(404, "Task not found.") from exc

    @app.post("/tasks/{task_id}/transition")
    def transition_task(task_id: str, req: TaskTransitionRequest):
        try:
            task = caretaker.transition(
                task_id,
                req.status,
                user_approved=req.user_approved,
                verification_evidence=tuple(req.verification_evidence),
            )
        except KeyError as exc:
            raise HTTPException(404, "Task not found.") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except (InvalidTransition, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return jsonable_encoder(task)

    @app.post("/tasks/{task_id}/steps")
    def record_task_step(task_id: str, req: TaskStepRequest):
        try:
            caretaker.record_step(task_id, req.step_id)
            return jsonable_encoder(caretaker.get_task(task_id))
        except KeyError as exc:
            raise HTTPException(404, "Task not found.") from exc
        except InvalidTransition as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/tasks/{task_id}/authorize")
    def authorize_task_action(task_id: str, req: TaskAuthorizationRequest):
        try:
            permitted = caretaker.authorize_action(
                task_id,
                consequential=req.consequential,
                user_approved=req.user_approved,
                policy_level=AutonomyLevel(settings.autonomy_level),
            )
        except KeyError as exc:
            raise HTTPException(404, "Task not found.") from exc
        return {"permitted": permitted}

    @app.get("/tasks/{task_id}/events")
    def task_events(task_id: str):
        try:
            caretaker.get_task(task_id)
        except KeyError as exc:
            raise HTTPException(404, "Task not found.") from exc
        events = [event for event in caretaker.events if event.task_id == task_id]
        return {"events": jsonable_encoder(events)}

    @app.delete("/tasks/{task_id}")
    def delete_task(task_id: str):
        if not caretaker.delete_task(task_id):
            raise HTTPException(404, "Task not found.")
        return {"message": "Task and its audit events were deleted."}

    @app.post("/feedback")
    def feedback(req: FeedbackRequest):
        text = f"Question: {req.question.strip()}\nCorrection: {req.correction.strip()}"
        try:
            item = memory_store.add(text, "correction")
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"message": "Thanks. I saved your correction for related questions.",
                "memory": item}

    @app.post("/read-file")
    async def read_file(file: UploadFile = File(...),
                        conversation_id: str | None = Form(default=None)):
        """Read an uploaded document and return a grounded summary."""
        filename = file.filename or ""
        try:
            data = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"File is too large. The limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
            summary, sources = extract_and_summarize(
                filename, data,
                OllamaClient(settings.ollama_host, settings.llm_model),
                keep_alive=settings.keep_alive)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except OllamaDown as e:
            raise HTTPException(503, f"Ollama is not reachable: {e}. Start the Ollama app.") from e
        except OllamaError as e:
            raise HTTPException(503, f"Ollama request failed: {e}") from e
        except Exception as e:
            raise HTTPException(422, f"Could not read {filename or 'the uploaded file'}: {e}") from e
        finally:
            await file.close()
        if conversation_id and storage.mongo_ready:
            storage.append_message(conversation_id, "user", f"Read this file: {filename}")
            storage.append_message(conversation_id, "assistant", summary, sources)
        return {"file": filename, "summary": summary, "sources": sources,
                "model": settings.llm_model, "conversation_id": conversation_id}

    @app.post("/query")
    def query(req: QueryRequest):
        history = _recent_history(storage, req.conversation_id)

        def save_turn(result: dict) -> dict:
            if req.conversation_id and storage.mongo_ready:
                if storage.get_conversation(req.conversation_id) is None:
                    raise HTTPException(404, "Conversation not found. Start a new chat and try again.")
                storage.append_message(req.conversation_id, "user", req.question)
                storage.append_message(req.conversation_id, "assistant", result["answer"],
                                       result.get("sources", []))
                return {**result, "conversation_id": req.conversation_id}
            return result

        if persona.is_self_question(req.question):
            return save_turn({"answer": persona.self_description(settings.llm_model), "sources": [],
                    "timing": {"retrieve_ms": 0, "generate_ms": 0,
                               "tokens": 0, "tok_per_s": 0.0}})
        reminder_answer = persona.reminder_limit_response(req.question)
        if reminder_answer is not None:
            return save_turn({"answer": reminder_answer, "sources": [],
                    "timing": {"retrieve_ms": 0, "generate_ms": 0,
                               "tokens": 0, "tok_per_s": 0.0}})
        remembered = parse_remember_command(req.question)
        if remembered:
            item = memory_store.add(remembered, inferred_kind(remembered))
            return save_turn({"answer": "Got it — I'll remember that.", "sources": [],
                    "memory": item, "timing": {"retrieve_ms": 0, "generate_ms": 0,
                                                 "tokens": 0, "tok_per_s": 0.0}})
        action_answer = run_browser_command(req.question, browser_opener)
        if action_answer is not None:
            return save_turn({"answer": action_answer, "sources": [],
                    "timing": {"retrieve_ms": 0, "generate_ms": 0,
                               "tokens": 0, "tok_per_s": 0.0}})
        scheduled_answer = appointment_schedule(req.question)
        if scheduled_answer is not None:
            return save_turn({"answer": scheduled_answer, "sources": [],
                    "timing": {"retrieve_ms": 0, "generate_ms": 0,
                               "tokens": 0, "tok_per_s": 0.0}})
        active_engine = current_engine()
        if active_engine is None:
            memories = memory_store.recall(req.question)
            terms = set(re.findall(r"\w+", req.question.casefold()))
            evidence = [item for item in memories if item["kind"] != "preference" or
                        terms & set(re.findall(r"\w+", item["text"].casefold()))]
            personal_question = persona.is_personal_fact_question(req.question)
            if evidence or not personal_question:
                context = "[Personal memory]\n" + "\n".join(
                    f"({item['kind']}) {item['text']}" for item in memories)
                try:
                    messages = [
                        {"role": "system", "content": persona.build_system_prompt(
                            context, grounded_only=personal_question)},
                        *history,
                        {"role": "user", "content": req.question},
                    ]
                    result = OllamaClient(settings.ollama_host, settings.llm_model).chat(
                        messages, think=should_use_deep_thinking(
                            req.question, req.deep_think, settings.llm_model),
                        max_tokens=settings.deep_think_max_tokens
                        if should_use_deep_thinking(
                            req.question, req.deep_think, settings.llm_model) else None,
                        keep_alive=settings.keep_alive)
                except OllamaDown as e:
                    raise HTTPException(503, f"Ollama is not reachable: {e}. Start the Ollama app.") from e
                except OllamaError as e:
                    raise HTTPException(503, f"Ollama request failed: {e}") from e
                return save_turn({"answer": result.text,
                        "sources": [{"file": "personal memory", "snippet": item["text"][:200],
                                     "score": 1.0} for item in evidence],
                        "timing": {"retrieve_ms": 0, "generate_ms": 0,
                                   "tokens": result.eval_count, "tok_per_s": result.tok_per_s}})
            return save_turn({"answer": persona.REFUSAL, "sources": [],
                    "timing": {"retrieve_ms": 0, "generate_ms": 0,
                               "tokens": 0, "tok_per_s": 0.0}})
        try:
            query_args = {"top_k": req.top_k, "deep_think": req.deep_think}
            if history:
                query_args["history"] = history
            with engine_lock:
                active_engine = engine or active_engine
                res = active_engine.query(req.question, **query_args)
        except OllamaDown as e:
            raise HTTPException(503, f"Ollama is not reachable: {e}. Start the Ollama app.")
        except OllamaError as e:
            raise HTTPException(503, f"Ollama request failed: {e}")
        except EmptyIndexError as e:
            raise HTTPException(409, str(e))
        return save_turn({"answer": res.answer,
                "sources": [{"file": s.file, "snippet": s.snippet, "score": s.score} for s in res.sources],
                "timing": res.timing})

    @app.post("/voice/tts")
    def voice_tts(req: TextToSpeechRequest):
        if tts is None:
            raise HTTPException(503, f"Voice output is unavailable: {tts_status}")
        try:
            audio = tts.synthesize(req.text.strip())
        except PiperError as e:
            raise HTTPException(503, f"Could not create voice reply: {e}") from e
        return Response(content=audio, media_type="audio/wav",
                        headers={"Content-Disposition": 'inline; filename="diya.wav"'})

    @app.post("/voice/transcribe")
    async def voice_transcribe(file: UploadFile = File(...)):
        """Transcribe a locally recorded 16-bit PCM WAV file with Vosk."""
        if stt is None:
            raise HTTPException(503, f"Voice input is unavailable: {stt_status}")
        try:
            audio = await file.read(MAX_AUDIO_BYTES + 1)
            if not audio:
                raise ValueError("The audio recording is empty.")
            if len(audio) > MAX_AUDIO_BYTES:
                raise ValueError("The recording is too large. The limit is 10 MB.")
            transcript = stt.transcribe(audio)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except VoskError as e:
            raise HTTPException(422, f"Could not transcribe this recording: {e}") from e
        finally:
            await file.close()
        if not transcript.text:
            raise HTTPException(422, "I could not hear any words. Please try recording again.")
        return {"text": transcript.text, "confidence": transcript.confidence}

    @app.post("/voice/ask")
    async def voice_ask(file: UploadFile = File(...),
                        conversation_id: str | None = Form(default=None),
                        deep_think: bool = Form(default=False)):
        """Transcribe speech locally, answer it, and include speech audio when available."""
        if stt is None:
            raise HTTPException(503, f"Voice input is unavailable: {stt_status}")
        try:
            audio = await file.read(MAX_AUDIO_BYTES + 1)
            if not audio:
                raise ValueError("The audio recording is empty.")
            if len(audio) > MAX_AUDIO_BYTES:
                raise ValueError("The recording is too large. The limit is 10 MB.")
            transcript = stt.transcribe(audio)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except VoskError as e:
            raise HTTPException(422, f"Could not transcribe this recording: {e}") from e
        finally:
            await file.close()
        if not transcript.text:
            raise HTTPException(422, "I could not hear any words. Please try recording again.")

        result = query(QueryRequest(question=transcript.text, deep_think=deep_think,
                                    conversation_id=conversation_id))
        audio_base64 = None
        if tts is not None:
            try:
                audio_base64 = base64.b64encode(tts.synthesize(result["answer"])).decode("ascii")
            except PiperError as e:
                raise HTTPException(503, f"Could not create voice reply: {e}") from e
        return {**result, "transcript": transcript.text,
                "transcription_confidence": transcript.confidence,
                "audio_format": "wav" if audio_base64 else None,
                "audio_base64": audio_base64}

    @app.post("/query/voice")
    def query_with_voice(req: QueryRequest):
        """Return Diya's usual answer and matching spoken audio."""
        if tts is None:
            raise HTTPException(503, f"Voice output is unavailable: {tts_status}")
        result = query(req)
        try:
            audio = tts.synthesize(result["answer"])
        except PiperError as e:
            raise HTTPException(503, f"Could not create voice reply: {e}") from e
        return {**result, "audio_format": "wav",
                "audio_base64": base64.b64encode(audio).decode("ascii")}

    @app.post("/query/stream")
    def query_stream(req: QueryRequest):
        history = _recent_history(storage, req.conversation_id)
        if persona.is_self_question(req.question):
            answer = persona.self_description(settings.llm_model)
            def self_sse():
                yield f"data: {json.dumps({'type': 'delta', 'text': answer}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'final', 'sources': [], 'timing': {'retrieve_ms': 0}}, ensure_ascii=False)}\n\n"
            return StreamingResponse(self_sse(), media_type="text/event-stream")
        reminder_answer = persona.reminder_limit_response(req.question)
        if reminder_answer is not None:
            def reminder_sse():
                yield f"data: {json.dumps({'type': 'delta', 'text': reminder_answer}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'final', 'sources': [], 'timing': {'retrieve_ms': 0}}, ensure_ascii=False)}\n\n"
            return StreamingResponse(reminder_sse(), media_type="text/event-stream")
        remembered = parse_remember_command(req.question)
        if remembered:
            memory_store.add(remembered, inferred_kind(remembered))
            answer = "Got it — I'll remember that."
            def remember_sse():
                yield f"data: {json.dumps({'type': 'delta', 'text': answer}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'final', 'sources': [], 'timing': {'retrieve_ms': 0}}, ensure_ascii=False)}\n\n"
            return StreamingResponse(remember_sse(), media_type="text/event-stream")
        action_answer = run_browser_command(req.question, browser_opener)
        if action_answer is not None:
            def action_sse():
                yield f"data: {json.dumps({'type': 'delta', 'text': action_answer}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'final', 'sources': [], 'timing': {'retrieve_ms': 0, 'generate_ms': 0, 'tokens': 0, 'tok_per_s': 0.0}}, ensure_ascii=False)}\n\n"
            return StreamingResponse(action_sse(), media_type="text/event-stream")
        scheduled_answer = appointment_schedule(req.question)
        if scheduled_answer is not None:
            def schedule_sse():
                yield f"data: {json.dumps({'type': 'delta', 'text': scheduled_answer}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'final', 'sources': [], 'timing': {'retrieve_ms': 0, 'generate_ms': 0, 'tokens': 0, 'tok_per_s': 0.0}}, ensure_ascii=False)}\n\n"
            return StreamingResponse(schedule_sse(), media_type="text/event-stream")
        active_engine = current_engine()
        if active_engine is None:
            memories = memory_store.recall(req.question)
            terms = set(re.findall(r"\w+", req.question.casefold()))
            evidence = [item for item in memories if item["kind"] != "preference" or
                        terms & set(re.findall(r"\w+", item["text"].casefold()))]
            personal_question = persona.is_personal_fact_question(req.question)
            if evidence or not personal_question:
                context = "[Personal memory]\n" + "\n".join(
                    f"({item['kind']}) {item['text']}" for item in memories)
                def memory_sse():
                    try:
                        for delta in OllamaClient(settings.ollama_host, settings.llm_model).chat_stream([
                            {"role": "system", "content": persona.build_system_prompt(
                                context, grounded_only=personal_question)},
                            *history,
                            {"role": "user", "content": req.question},
                        ], think=should_use_deep_thinking(
                            req.question, req.deep_think, settings.llm_model),
                            max_tokens=settings.deep_think_max_tokens
                            if should_use_deep_thinking(
                                req.question, req.deep_think, settings.llm_model) else None,
                            keep_alive=settings.keep_alive):
                            yield f"data: {json.dumps({'type': 'delta', 'text': delta}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps({'type': 'final', 'sources': [{'file': 'personal memory', 'snippet': item['text'][:200], 'score': 1.0} for item in evidence], 'timing': {'retrieve_ms': 0}}, ensure_ascii=False)}\n\n"
                    except OllamaError as e:
                        yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"
                return StreamingResponse(memory_sse(), media_type="text/event-stream")
            def refusal_sse():
                yield f"data: {json.dumps({'type': 'delta', 'text': persona.REFUSAL}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'final', 'sources': [], 'timing': {'retrieve_ms': 0}}, ensure_ascii=False)}\n\n"
            return StreamingResponse(refusal_sse(), media_type="text/event-stream")
        def sse():
            try:
                stream_args = {"top_k": req.top_k, "deep_think": req.deep_think}
                if history:
                    stream_args["history"] = history
                with engine_lock:
                    stream_engine = engine or active_engine
                    for event in stream_engine.query_stream(req.question, **stream_args):
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except OllamaError as e:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return app


def main() -> None:
    import uvicorn
    settings = get_settings()
    uvicorn.run(create_app(settings=settings), host="127.0.0.1", port=settings.api_port)


if __name__ == "__main__":
    main()
