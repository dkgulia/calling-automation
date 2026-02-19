# TODO: Replace with real STT provider (e.g. Deepgram, AssemblyAI).
#
# In production this will:
#   1. Accept raw audio frames from Pipecat transport
#   2. Stream them to the STT service
#   3. Return interim and final transcripts
#
# Integration point for Pipecat:
#   pipeline.add(DeepgramSTTService(api_key=settings.stt_api_key))


async def transcribe(audio_chunk: bytes) -> str:
    """Stub: pretends to transcribe audio."""
    return "[stub transcript from audio]"
