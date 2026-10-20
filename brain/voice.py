class VoiceError(RuntimeError):
    pass


class PiperError(VoiceError):
    pass


class VoskError(VoiceError):
    pass


def resolve_tts(settings):
    """Load configured text-to-speech engine, returning engine and status."""
    if settings.tts_engine.casefold() == "off":
        return None, "off"
    if settings.tts_engine.casefold() != "auto":
        return None, f"unsupported BRAIN_TTS_ENGINE value: {settings.tts_engine}"
    try:
        from brain.piper import PiperEngine
        return PiperEngine.discover(settings), "ready"
    except PiperError as exc:
        return None, str(exc)


def resolve_stt(settings):
    if settings.stt_engine.casefold() == "off":
        return None, "off"
    if settings.stt_engine.casefold() != "auto":
        return None, f"unsupported BRAIN_STT_ENGINE value: {settings.stt_engine}"
    try:
        from brain.vosk_stt import VoskEngine
        return VoskEngine.discover(settings), "ready"
    except VoskError as exc:
        return None, str(exc)
