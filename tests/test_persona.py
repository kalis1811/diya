from brain import persona


def test_persona_has_female_name():
    assert persona.NAME == "Diya"
    assert persona.NAME in persona.PERSONA_PROMPT
    assert persona.NAME in persona.INTRO


def test_persona_uses_her_name_when_asked():
    p = persona.PERSONA_PROMPT.lower()
    assert 'when the user asks who you are, say "i am diya"' in p


def test_persona_is_kind_and_warm():
    p = persona.PERSONA_PROMPT.lower()
    assert "kind" in p
    assert "warm" in p


def test_kindness_never_overrides_grounding():
    prompt = persona.build_system_prompt("ctx", grounded_only=True)
    assert "Answer only from the CONTEXT below" in prompt
    assert persona.REFUSAL in prompt
    assert "never invent facts" in prompt
