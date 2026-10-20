from pathlib import Path

from brain.chat_exports import detect_and_parse, parse_telegram_json, parse_whatsapp

FIX = Path(__file__).parent / "fixtures"


def test_parse_whatsapp_android_multiline_and_media():
    msgs = parse_whatsapp(FIX / "chat_sample_android.txt")
    assert len(msgs) == 4  # media line is a continuation of Alice's message
    assert msgs[0].sender == "Alice"
    assert msgs[0].text == "hey are you coming tomorrow?"
    assert msgs[1].sender == "Bob"
    assert "depends on when it starts tbh" in msgs[1].text
    assert "<Media omitted>" in msgs[2].text
    assert msgs[3].sender == "Bob"
    assert msgs[3].text == "ok see you there 👍"  # unicode round-trip


def test_parse_whatsapp_ios_bracket_format():
    msgs = parse_whatsapp(FIX / "chat_sample_ios.txt")
    assert len(msgs) == 2
    assert msgs[1].sender == "Bob"
    assert msgs[1].text == "got it, reviewing now"


def test_parse_telegram_json_flattens_rich_text():
    msgs = parse_telegram_json(FIX / "telegram_result.json")
    assert len(msgs) == 2  # service messages skipped
    assert msgs[0].sender == "Alice"
    assert msgs[1].text == "yes here done"


def test_detect_and_parse_builds_documents_with_metadata():
    docs = detect_and_parse(FIX / "telegram_result.json")
    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["category"] == "chats"
    assert meta["app"] == "telegram"
    assert meta["chat_name"] == "Project Chat"
    assert meta["file_name"] == "telegram_result.json"
    assert len(meta["file_hash"]) == 64  # sha256 hex
    assert "Alice: did you finish the report?" in docs[0].text


def test_non_utf8_chat_file_is_tolerated(tmp_path):
    p = tmp_path / "legacy.txt"
    p.write_bytes("15/02/26, 09:14 - Alice: caf\xe9 run\n".encode("latin-1"))
    docs = detect_and_parse(p)
    assert len(docs) == 1
    assert "caf" in docs[0].text  # decoded with replacement, not a crash


def test_date_range_is_chronological_across_months(tmp_path):
    p = tmp_path / "cross.txt"
    p.write_text(
        "31/01/26, 23:00 - Alice: see you feb\n"
        "01/02/26, 08:00 - Bob: morning!\n", encoding="utf-8")
    docs = detect_and_parse(p)
    assert docs[0].metadata["date_range"] == "31/01/26, 23:00 .. 01/02/26, 08:00"
