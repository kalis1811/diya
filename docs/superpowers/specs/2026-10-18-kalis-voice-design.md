# Voice Engines Design (Piper TTS + Vosk STT)

Date: 2026-10-18
Status: approved in chat, implementation starting
Input: prior local evaluation of Piper text-to-speech and Vosk speech recognition.
Voice assets are optional and are not bundled with the repository.

## Goal

The user talks to Diya by voice: they record a question (any recorder app) and POST
the WAV; Diya answers and the answer comes back as WAV audio for the speakers.
Full loop, API-only surface — no web UI, no PyAudio, no live mic streaming.

## Decisions (user-approved)

- **Full loop**: Vosk transcribes the question; Piper speaks Diya's answer.
- **API only**: the user records WAV files themselves; no browser mic capture.
- **Artifacts**: copied from a user-provided local asset directory into this
  repo's gitignored `tools/` and `storage/models/` by `scripts/setup_voice.py`;
  a fresh checkout can supply assets downloaded from the
  official sources: GitHub `rhasspy/piper` release, HuggingFace
  `rhasspy/piper-voices`, alphacephei.com model list).
- **Rejected alternatives**: `piper-tts` pip package (build-fragile on Windows);
  integrated speech servers (LocalAI/Wyoming — overkill); SAPI fallback
  (voice is optional in KALIS, so a missing engine is a loud 503, never a
  silent robotic downgrade).

## Components

### `brain/piper.py` — PiperEngine (TTS)

- Constructed with explicit `exe` and `voice` paths; `discover(settings)`
  resolves: exe = `BRAIN_PIPER_EXE` override → `<repo>/tools/piper/piper.exe`;
  voice = glob `<repo>/storage/models/piper/*.onnx` requiring the `.onnx.json`
  sidecar, preferring `BRAIN_PIPER_VOICE` (`en_US-amy-medium`).
- `synthesize(text) -> bytes`: text on stdin (utf-8), `-m <voice.onnx> -f
  <temp.wav>`, reads and returns the WAV bytes, deletes the temp file.
- The subprocess runner is injectable (`run(cmd, stdin_bytes) -> (code, stderr)`)
  so unit tests need no real exe.
- Nonzero exit or missing files → `PiperError` quoting stderr and naming
  `uv run python scripts/setup_voice.py`.

### `brain/vosk_stt.py` — VoskEngine (STT)

- Model dir = `BRAIN_VOSK_MODEL` (name under `<repo>/storage/models/vosk/` or an
  absolute path); model loads lazily on first transcribe, cached behind a lock.
- `transcribe(wav_bytes) -> Transcript(text, confidence)`: `wave.open` over
  BytesIO, stereo→mono downmix with `array` (no `audioop`), `KaldiRecognizer`
  over 4000-frame chunks, word-level confidences averaged.
- The recognizer factory is injectable for unit tests.
- Missing model → `VoskError`; undecodable WAV → `VoskError`.
- `vosk.SetLogLevel(-1)` silences Kaldi's native stderr.

### `brain/voice.py` — resolution

- `VoiceError` base (`PiperError`/`VoskError` subclass it).
- `resolve_tts(settings)` / `resolve_stt(settings)` → engine instance or `None`
  with a human-readable reason. `BRAIN_TTS_ENGINE`/`BRAIN_STT_ENGINE`:
  `auto` (discover; missing → disabled) or `off`. No other values.
- Resolvers import engines lazily (no import cycle; engines import only
  `VoiceError` from this module).

### API ([brain/api.py](../../../brain/api.py) extensions)

- `create_app(engine=None, settings=None, tts=None, stt=None)` — voice engines
  injectable like the query engine; when not injected they resolve from settings.
- `POST /voice/tts` — `{"text": str (min_length=1)}` → `audio/wav` bytes.
- `POST /voice/stt` — multipart WAV upload → `{"text", "confidence"}`.
  400 if undecodable, 413 over 25 MB.
- `POST /voice/ask` — the loop: WAV upload → transcript → `engine.query()`
  (same error mapping as `/query`: OllamaDown→503, EmptyIndexError→409)
  → Piper synth of Diya's answer → `{"transcript", "answer", "sources",
  "timing", "audio_base64"}`. `?deep_think=true` passes through.
- Missing engine → 503 naming the setting and the setup command; validation
  failures are FastAPI's standard 422.
- `GET /health` gains `voice: {"tts": <state>, "stt": <state>}` where state is
  `"ready"`, `"off"`, or the disable reason — existing keys unchanged.

### Config ([brain/config.py](../../../brain/config.py) `_DEFAULTS`)

`BRAIN_TTS_ENGINE=auto`, `BRAIN_STT_ENGINE=auto`, `BRAIN_PIPER_EXE=` (empty =
auto-discover), `BRAIN_PIPER_VOICE=en_US-amy-medium`,
`BRAIN_VOSK_MODEL=vosk-model-small-en-us-0.15`.
One new dependency: `vosk` (uv). `.gitignore` gains `tools/` and `storage/`.

### `scripts/setup_voice.py`

Copies piper.exe + DLLs + `espeak-ng-data`, the amy voice pair, and the vosk
model from the required `--from` source directory into `tools/` and
`storage/models/`, then verifies every expected file exists and is non-empty,
printing byte sizes. Missing source → nonzero exit naming what was not found.

## Data flow

WAV bytes → Vosk (CPU/RAM only) → transcript → RRF evidence gate →
qwen3.5:4b (VRAM, untouched) → Diya's answer (refusals included — she says
them kindly) → Piper (CPU) → WAV → base64 in the JSON response.

## Testing

- Unit (no artifacts required): discovery failures name the setup command;
  downmix math; lazy model load once; API contracts with injected fakes;
  25 MB cap; setup-script copy + verification against a fake source tree.
- Live (`tests/test_voice_live.py`, skipif artifacts absent): real Piper synth
  → WAV header (RIFF, 22050 Hz) → Vosk roundtrip (≥2 words) → full `/voice/ask`
  when Ollama is up.
- TDD throughout; `uv` only; full offline suite green before every commit.

## Non-goals

Web UI, live mic streaming, PyAudio, ffmpeg, SAPI, streaming TTS (Diya speaks
the finished answer), multi-language, voice memory integration (Phase 2).
