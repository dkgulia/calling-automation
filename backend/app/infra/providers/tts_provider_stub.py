# TODO: Replace with real TTS provider (e.g. ElevenLabs, PlayHT).
#
# In production this will:
#   1. Accept text from the LLM response
#   2. Stream it to the TTS service
#   3. Return audio frames to Pipecat transport for playback
#
# Integration point for Pipecat:
#   pipeline.add(ElevenLabsTTSService(api_key=settings.tts_api_key))


async def synthesize(text: str) -> bytes:
    """Stub: pretends to synthesize speech."""
    return b"[stub audio bytes]"
