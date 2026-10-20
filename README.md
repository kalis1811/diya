# Personal AI Brain

A local assistant for general questions, practical help, and reasoning that can
also answer from files you provide. Documents, embeddings, memories, and
generated answers stay on this machine. The API listens on
`127.0.0.1` by default.

## Requirements

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) installed and running locally
- `phi4-mini:latest` for answers and `qwen3-embedding:0.6b` for retrieval
- Docker Desktop is optional; without MongoDB, Diya uses local memory fallback

For the Windows desktop client, also install Flutter and Visual Studio with the
**Desktop development with C++** workload. The web UI can run without Flutter.

Clone the repository, open a terminal in its folder, then install dependencies
and the two local models:

```powershell
uv sync
ollama pull phi4-mini:latest
ollama pull qwen3-embedding:0.6b
uv run python scripts/setup_admin.py
uv run python -m brain.ingest
uv run python -m brain.api
```

`setup_admin.py` saves your chosen single-owner credentials in the ignored
`.env` file. Check Ollama and both configured models with
`uv run python scripts/check_models.py`. Open `http://127.0.0.1:8400/` for the
web UI. MongoDB is optional for chat history; memory has a local fallback.
To save persistent chat history, start MongoDB in another terminal with
`docker compose up -d mongodb` before launching the API.

## Add knowledge

Add and manage documents in the app with **Knowledge base** in the sidebar. You
can upload a file, see the saved local library and indexed chunk count, and
remove a file; changes are indexed immediately. This accepts PDF, TXT,
Markdown, CSV, DOCX, HTML, and JSON files (up to 25 MB each). A regular chat
attachment still gives a one-time summary; use **Add to knowledge** on the
selected file or the Knowledge base page to keep it available for future
questions.

For bulk imports, place source files in the matching folders:

- `data/raw/docs/` — notes and documents
- `data/raw/web/` — saved web pages and clippings
- `data/raw/chats/` — WhatsApp text exports or Telegram JSON exports

Build or update the local search index:

```powershell
uv run python -m brain.ingest
```

The ingester adds new and changed files, removes chunks from deleted or emptied
files, and keeps the previous indexed version if a changed file cannot be read.
Use `--rebuild` to recreate the entire index. Stop the API before running the
CLI ingester or rebuilding because embedded Qdrant allows only one process to
open the index. In-app additions and removals coordinate with active queries
and refresh the index automatically.

## Ask questions

Start MongoDB first. With Docker Desktop installed, open a PowerShell window
in the project folder and run:

```powershell
uv sync
docker compose up -d mongodb
uv run python -m brain.api
```

On Windows, `run_diya.bat` prepares dependencies, asks you to set the admin
login on first run, checks Ollama and the configured models, indexes files
already in `data/raw/`, starts MongoDB when Docker is available, launches the
API, and builds/opens the Flutter desktop client. Keep the API terminal open.
The launcher prints the exact `ollama pull` commands if a required model is
missing. You can also use the web UI at `http://127.0.0.1:8400/`.

Diya's clients use a single-owner admin login. Choose a private password of at
least 10 characters. The password is saved only in the ignored local `.env`
file and is never printed. The admin username and password protect the UI and API;
`/health` remains available to the local launcher. The API binds to
`127.0.0.1`, so it is not exposed to other devices on your network.

Check readiness at `http://127.0.0.1:8400/health`, then send a question. General
questions can be answered even before files are indexed; personal facts from
your files still need ingestion. Open `http://127.0.0.1:8400/` for the web app.
It saves conversations and user-approved memories in the local MongoDB database
named `diya`. MongoDB data lives in the `diya_mongodb` Docker volume. The existing
Qdrant knowledge index and original RAG files remain in the local `data/` folder.
If MongoDB is stopped, the app reports that state and uses the existing local
memory file as a fallback; saved chat history is available again when MongoDB
reconnects.

To stop MongoDB without deleting saved data, run `docker compose stop mongodb`.
Do not use `docker compose down -v` unless you intend to erase the MongoDB data.

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8400/query `
  -Method Post -ContentType 'application/json' `
  -Body '{"question":"What is in my notes?"}'
```

Upload a document to get a summary immediately, without adding it to the
long-term RAG index. The `/read-file` endpoint accepts PDFs (including scanned
PDFs when OCR is available), images such as JPEG and PNG (OCR), and formats
supported by the document readers such as TXT, Markdown, DOCX, CSV, and HTML.
For example, from PowerShell:

```powershell
curl.exe -F "file=@C:\path\to\report.pdf" http://127.0.0.1:8400/read-file
```

The response contains a summary and page or file sources. Files sent to
`/read-file` are processed temporarily. To keep one available for future
questions, use the Knowledge base page or put it in `data/raw/docs/` and run the
ingestion command above.

For a one-shot terminal answer, use:

```powershell
uv run python scripts/ask.py "What is in my notes?"
```

## Optional voice support

Voice input and output require local Vosk and Piper binaries/models. They are
not bundled in this repository. If you already have the expected assets in the
layout described by `scripts/setup_voice.py`, copy them into this checkout with:

```powershell
uv run python scripts/setup_voice.py --from C:\path\to\voice-assets
```

Restart the API after setup and open `http://127.0.0.1:8400/` in a browser.
The chat page shows the readable answer, sources, and plays the spoken answer
when voice is enabled. A play control is available if the browser blocks automatic
playback. `/health` reports whether voice output is ready.

The page uses `/query/voice` to get both the readable answer and WAV audio. For
PowerShell-only use, send a question and save/play the spoken reply:

```powershell
$reply = Invoke-RestMethod -Uri http://127.0.0.1:8400/query/voice `
  -Method Post -ContentType 'application/json' `
  -Body '{"question":"Explain photosynthesis in simple words."}'
$wav = Join-Path $PWD 'diya-reply.wav'
[IO.File]::WriteAllBytes($wav, [Convert]::FromBase64String($reply.audio_base64))
$reply.answer
Start-Process $wav
```

The regular `/query` endpoint still returns text. `/voice/tts` can also turn
any supplied text into a WAV file directly.

The same query interface recognizes explicit commands such as `open YouTube`,
`open Google`, `search Google for ...`, and `search YouTube for ...`. It opens
only Google or YouTube in the computer running the local API. This opens public
pages and search results; it does not sign in to a Google account.

When you ask about a personal fact, answers use matching files or saved memory
and cite those sources. If the information is not available, the assistant says
so instead of guessing. For general questions and practical help, it can reason
from its model knowledge even when the files contain no relevant evidence.

## Learning, thinking, and self-understanding

Teach the assistant a fact or preference with a normal request such as:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8400/query `
  -Method Post -ContentType 'application/json' `
  -Body '{"question":"Remember that I prefer simple explanations."}'
```

Approved facts and corrections are stored in local MongoDB when it is available,
or in `data/memory.json` as a fallback, and can be used in related future
answers. Preferences can guide answers generally. Use
`GET /memory` to review saved items, `POST /feedback` to correct an answer, and
`DELETE /memory/{id}` to forget an item. The assistant does not silently train
or change Qwen's model weights; learning happens through this user-controlled
memory. For example, send feedback with
`POST /feedback` and a JSON body like
`{"question":"Where do I study?","correction":"I study at Example University."}`.

Deep thinking is off by default to keep ordinary answers faster and reduce
token use. Turn on **Use extra thinking** in the chat page when a question needs
more careful reasoning. You can also pass `"deep_think": true` with a query or
use `scripts/ask.py --deep`. Ask “Who are you?” or open `GET /about` to see what the assistant
can do and its limits. It is software powered by the configured local Ollama
model, not a conscious being. For MongoDB-backed chats, it uses only a bounded
recent window of conversation history.

## Brain workflow

Diya handles each request locally in this order: receive text, speech, or a file;
transcribe speech or extract/understand file contents; combine the request with
recent conversation turns, relevant memories, and retrieved documents; apply the
personal-fact grounding rules; then ask the configured Ollama model to respond.
When text-to-speech is available, Diya can return the spoken answer as well.
Uploaded documents are source material, not instructions that override the
user's request or Diya's rules.

For MongoDB-backed conversations, up to the latest eight messages are supplied
as context, and short follow-up questions can use the previous user question to
guide retrieval. When MongoDB is unavailable, the browser keeps the current chat
locally, but those turns are not available to the brain API as conversation
history.

Speech input is available through the local API. `/voice/transcribe` returns a
transcript; `/voice/ask` transcribes, answers, and includes spoken audio when
Piper is ready. Both accept a 16-bit PCM WAV recording. Image uploads use OCR
when available and send the image to the configured local Ollama model for
visual understanding; that model must support image input.

The development order is brain capabilities and reliability first, model
evaluation second, and UI/UX last. See [the brain workflow](docs/brain-workflow.md)
for the implementation and validation sequence.

## Current scope

The current RAG index is local and project-specific. Chat imports support
WhatsApp text and Telegram JSON exports. Other import formats, Google account
access, automatic folder watching, and a personalized fine-tuned model are not
implemented. Voice recording is available in the Flutter client when local
voice assets are installed.

## V2: prepare a personal-style dataset

The first V2 step prepares WhatsApp TXT and Telegram JSON exports for a future
local fine-tune. Put exports in `data/raw/chats/`, then run this command. For the
WhatsApp exports currently supplied, the sender label is `You`:

```powershell
uv run python scripts/prepare_finetune_data.py --user-name "You"
```

The script writes `all.jsonl` (every example), `train.jsonl`, `validation.jsonl`,
and `stats.md` to `data/finetune/`. That folder is ignored by Git. `all.jsonl`
contains every example from the exports currently in `data/raw/chats/`; `stats.md`
shows the latest counts. Review the examples locally before training, especially
the sender labels and private content. This step does not train or change the
assistant model. A local fine-tuning runner and evaluation gate are still needed
before the assistant can use a personalized model.
