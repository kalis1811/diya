import argparse
import re

from brain import persona
from brain.browser_actions import run_browser_command
from brain.config import get_settings
from brain.engine import EmptyIndexError, load_engine, should_use_deep_thinking
from brain.memory import inferred_kind, parse_remember_command
from brain.ollama_client import OllamaClient
from brain.storage import get_storage
from brain.scheduling import appointment_schedule


def _answer_without_index(question: str, settings, deep_think: bool) -> str:
    if persona.is_self_question(question):
        return persona.self_description(settings.llm_model)
    remembered = parse_remember_command(question)
    if remembered:
        get_storage(settings).memories.add(remembered, inferred_kind(remembered))
        return "Got it — I'll remember that."
    action = run_browser_command(question)
    if action is not None:
        return action
    scheduled = appointment_schedule(question)
    if scheduled is not None:
        return scheduled

    memories = get_storage(settings).memories.recall(question)
    terms = set(re.findall(r"\w+", question.casefold()))
    evidence = [item for item in memories if item["kind"] != "preference" or
                terms & set(re.findall(r"\w+", item["text"].casefold()))]
    personal = persona.is_personal_fact_question(question)
    if personal and not evidence:
        return persona.REFUSAL
    context = "[Personal memory]\n" + "\n".join(
        f"({item['kind']}) {item['text']}" for item in memories)
    use_thinking = should_use_deep_thinking(question, deep_think, settings.llm_model)
    response = OllamaClient(settings.ollama_host, settings.llm_model).chat([
        {"role": "system", "content": persona.build_system_prompt(
            context, grounded_only=personal)},
        {"role": "user", "content": question},
    ], think=use_thinking,
        max_tokens=settings.deep_think_max_tokens if use_thinking else None,
        keep_alive=settings.keep_alive)
    return response.text


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask your brain a question")
    ap.add_argument("question")
    ap.add_argument("--deep", action="store_true", help="enable thinking mode (slower)")
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()
    settings = get_settings()
    try:
        engine = load_engine(settings)
    except EmptyIndexError:
        print(_answer_without_index(args.question, settings, args.deep))
        return
    res = engine.query(args.question, top_k=args.top_k, deep_think=args.deep)
    print(res.answer)
    if res.sources:
        print("\nSources:")
        for s in res.sources:
            print(f"  [{s.score:.2f}] {s.file}: {s.snippet[:100]}...")
    t = res.timing
    print(f"\n({t['retrieve_ms']}ms retrieve, {t['generate_ms']}ms generate, "
          f"{t['tokens']} tokens @ {t['tok_per_s']} tok/s)")


if __name__ == "__main__":
    main()
