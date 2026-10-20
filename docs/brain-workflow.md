# Diya brain workflow

Diya's brain runs locally. Keep the configured language model at or below the
4B target by default, and keep user files, conversation context, and model
calls on the local machine.

## Request flow

1. **Receive the request.** Text requests go to `/query`. Voice requests can
   go to `/voice/ask` as 16-bit PCM WAV audio. File uploads go to `/read-file`.
2. **Understand the input.** Vosk transcribes speech locally. Document readers
   extract text, OCR handles scanned pages and images when installed, and the
   configured Ollama model receives image uploads for visual understanding.
3. **Gather context.** For a MongoDB-backed chat, Diya adds the latest eight
   messages. Relevant personal memories and retrieved local documents are added
   as context. Short referential follow-ups can include the previous user turn
   in the retrieval query.
4. **Apply grounding and safety rules.** Personal facts must come from local
   evidence. Uploaded content and prior conversation are context, not policy
   instructions. Do not claim an action was taken unless a tool confirms it.
5. **Generate the response.** Ollama uses `BRAIN_LLM_MODEL` (default
   `phi4-mini:latest`). Voice requests also receive Piper audio when TTS is ready.
6. **Save the turn.** MongoDB stores chat history when connected. The browser
   fallback does not provide those local turns back to the backend as model
   context.

## Development order

1. Complete brain input, memory, retrieval, grounding, and response behavior.
2. Evaluate the configured 4B model on caretaker tasks; tune prompts or data
   only when repeatable evaluation shows a specific gap.
3. Validate the local speech and image paths.
4. Build microphone and image interaction controls in the UI after the brain
   endpoints and response contracts are stable.

## Test workflow

Run the application tests with `uv run pytest`. Pytest is configured to collect
from `tests/`, so training and model utility scripts under `scripts/` are not
mistaken for tests. Live Ollama tests run only when both configured models are
installed and the local Ollama API is reachable; otherwise they are skipped.

Keep offline tests independent of personal data and live MongoDB. Use temporary
indexes and isolated memory storage for test cases. Before changing an expected
result, check that the product behavior changed intentionally; do not weaken a
grounding or safety expectation just to make the test pass.
