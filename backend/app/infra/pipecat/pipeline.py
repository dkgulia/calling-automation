"""
Pipecat pipeline factory.

Creates a real-time voice pipeline for a single WebSocket session:
  transport.input() -> Deepgram STT -> BrainProcessor -> Cartesia TTS -> transport.output()

Each call to run_pipeline() blocks until the pipeline completes (EndFrame
pushed by BrainProcessor when the call ends, or client disconnect).

Barge-in: allow_interruptions=True lets user speech interrupt agent TTS.

Bot-ready handshake: PipecatClient expects an RTVI "bot-ready" message as a
protobuf Frame (field 4 = MessageFrame) sent as binary. Python's pipecat proto
only defines fields 1-3, so we manually encode the protobuf binary for field 4.
"""

from __future__ import annotations

import json

from fastapi import WebSocket

from deepgram import LiveOptions

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.core.logging import logger
from app.core.settings import settings
from app.infra.pipecat.brain import BrainProcessor
from app.usecases.run_simulation import OPENER


# RTVI bot-ready JSON payload
_BOT_READY_JSON = json.dumps({
    "label": "rtvi-ai",
    "type": "bot-ready",
    "id": "bot-ready",
    "data": {"version": "0.3.0", "config": []},
})


def _encode_varint(value: int) -> bytes:
    """Encode an integer as a protobuf varint."""
    parts = []
    while value > 0x7F:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    parts.append(value & 0x7F)
    return bytes(parts)


def _encode_bot_ready_protobuf() -> bytes:
    """
    Encode bot-ready as a protobuf Frame binary message.

    The JS client's proto has field 4 (MessageFrame) which the Python proto
    lacks. We manually encode:

      Frame {
        message (field 4, LEN) {
          data (field 1, LEN) = <bot-ready JSON string>
        }
      }

    Wire format:
      0x22 <varint:inner_len> 0x0a <varint:json_len> <json_bytes>
    """
    json_bytes = _BOT_READY_JSON.encode("utf-8")
    # Inner: MessageFrame.data (field 1, wire type 2 = LEN)
    inner = b"\x0a" + _encode_varint(len(json_bytes)) + json_bytes
    # Outer: Frame.message (field 4, wire type 2 = LEN)
    return b"\x22" + _encode_varint(len(inner)) + inner


# Pre-compute the binary bot-ready frame
_BOT_READY_BYTES = _encode_bot_ready_protobuf()


def _encode_text_frame(text: str) -> bytes:
    """Encode text as a protobuf Frame.text (field 1) → TextFrame.text (field 3)."""
    text_bytes = text.encode("utf-8")
    # TextFrame.text = field 3, wire type 2 (LEN)
    inner = b"\x1a" + _encode_varint(len(text_bytes)) + text_bytes
    # Frame.text = field 1, wire type 2 (LEN)
    return b"\x0a" + _encode_varint(len(inner)) + inner


def _encode_transcription_frame(text: str) -> bytes:
    """Encode text as a protobuf Frame.transcription (field 3) → TranscriptionFrame.text (field 3)."""
    text_bytes = text.encode("utf-8")
    # TranscriptionFrame.text = field 3, wire type 2 (LEN)
    inner = b"\x1a" + _encode_varint(len(text_bytes)) + text_bytes
    # Frame.transcription = field 3, wire type 2 (LEN)
    return b"\x1a" + _encode_varint(len(inner)) + inner


class _TranscriptSender(FrameProcessor):
    """Sends transcript text to the client WebSocket for live display.

    Use capture="user" to intercept TranscriptionFrame (user speech from STT).
    Use capture="agent" to intercept TTSSpeakFrame (agent response from brain).
    All frames are passed through unchanged.
    """

    def __init__(self, websocket: WebSocket, session_id: str, capture: str, **kwargs):
        super().__init__(**kwargs)
        self._ws = websocket
        self._sid = session_id
        self._capture = capture  # "user" or "agent"

    async def process_frame(self, frame, direction: FrameDirection):
        try:
            if self._capture == "user" and isinstance(frame, TranscriptionFrame) and frame.text.strip():
                data = _encode_transcription_frame(frame.text)
                await self._ws.send_bytes(data)
                logger.debug("Session %s: sent user transcript: %s", self._sid, frame.text[:60])
            elif self._capture == "agent" and isinstance(frame, TTSSpeakFrame) and frame.text.strip():
                data = _encode_text_frame(frame.text)
                await self._ws.send_bytes(data)
                logger.debug("Session %s: sent agent text: %s", self._sid, frame.text[:60])
        except Exception as exc:
            logger.warning("Session %s: failed to send transcript: %s", self._sid, exc)
        await self.push_frame(frame, direction)


async def run_pipeline(websocket: WebSocket, session_id: str) -> None:
    """
    Build and run a Pipecat voice pipeline for the given session.

    Blocks until the pipeline completes (EndFrame or client disconnect).
    Called directly from the /ws WebSocket endpoint handler.
    """
    logger.info("Session %s: building Pipecat pipeline", session_id)

    # --- Transport ---
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_audio_passthrough=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # --- STT: Deepgram ---
    stt = DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        live_options=LiveOptions(
            model="nova-3",
            language="en",
            punctuate=True,
            smart_format=True,
            interim_results=True,
            vad_events=True,
            utterance_end_ms="1500",
        ),
    )

    # --- Brain: domain layer bridge ---
    brain = BrainProcessor(session_id=session_id)

    # --- TTS: Cartesia ---
    tts = CartesiaTTSService(
        api_key=settings.cartesia_api_key,
        voice_id=settings.cartesia_voice_id,
    )

    # --- Transcript senders: forward user/agent text to client for live display ---
    # After STT: captures user speech (TranscriptionFrame only)
    user_transcript = _TranscriptSender(websocket=websocket, session_id=session_id, capture="user")
    # After brain: captures agent replies (TTSSpeakFrame only)
    agent_transcript = _TranscriptSender(websocket=websocket, session_id=session_id, capture="agent")

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_transcript,
        brain,
        agent_transcript,
        tts,
        transport.output(),
    ])

    # --- Barge-in: allow user speech to interrupt agent TTS ---
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
        ),
    )

    logger.info(
        "Session %s: barge-in enabled (allow_interruptions=True)",
        session_id,
    )

    # --- Send bot-ready + opener when client connects ---
    @transport.event_handler("on_client_connected")
    async def on_connected(transport_obj, client):
        # Send RTVI bot-ready as binary protobuf Frame (field 4 = MessageFrame).
        # Python's pipecat proto lacks this field, so we use pre-encoded bytes.
        # The JS ProtobufFrameSerializer.deserialize() expects binary Blob data,
        # not JSON text — so we MUST send as binary.
        logger.info("Session %s: sending bot-ready as protobuf binary", session_id)
        await websocket.send_bytes(_BOT_READY_BYTES)
        logger.info("Session %s: bot-ready sent, queueing opener TTS", session_id)
        await task.queue_frame(TTSSpeakFrame(text=OPENER))

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport_obj, client):
        logger.info("Session %s: client disconnected", session_id)

    # --- Run until complete ---
    runner = PipelineRunner()
    await runner.run(task)

    logger.info("Session %s: pipeline finished", session_id)
