"""
BrainProcessor — bridges Pipecat's frame pipeline to the domain layer.

Sits between Deepgram STT (upstream) and Cartesia TTS (downstream) in the
Pipecat pipeline.  When a finalized TranscriptionFrame arrives:
  1. Extracts the transcribed text
  2. Calls process_input(session_id, text) — async (Phase 4)
  3. Pushes a TTSSpeakFrame with the agent_text downstream to TTS
  4. If the call ended (action=END/CLOSE), pushes an EndFrame to tear down the pipeline

This processor contains NO domain logic — it is purely an adapter between
Pipecat's frame-based streaming model and the existing turn-based domain layer.
"""

from __future__ import annotations

import time

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.core.logging import logger
from app.usecases.process_input import process_input


class BrainProcessor(FrameProcessor):
    """Converts finalized speech transcriptions into agent voice responses."""

    def __init__(self, session_id: str, **kwargs):
        super().__init__(**kwargs)
        self._session_id = session_id
        self._processing = False  # guard against overlapping turns

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Drop interim transcriptions — they extend TextFrame and would be
        # spoken by TTS if allowed through.  Only final TranscriptionFrame
        # should reach the processing logic below.
        if isinstance(frame, InterimTranscriptionFrame):
            return

        if not isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)
            return

        text = frame.text.strip()
        if not text:
            return

        # Guard: skip if we are already processing a turn
        if self._processing:
            logger.warning(
                "Session %s: skipping overlapping transcription: %s",
                self._session_id,
                text[:80],
            )
            return

        self._processing = True
        try:
            logger.info("Session %s STT final: \"%s\"", self._session_id, text)

            t0 = time.monotonic()
            result = await process_input(self._session_id, text)
            brain_ms = (time.monotonic() - t0) * 1000

            agent_text = result.get("agent_text") or ""
            ended = result.get("ended", False)

            logger.info(
                "Session %s brain: text_len=%d ended=%s total=%.0fms",
                self._session_id,
                len(text),
                ended,
                brain_ms,
            )

            if agent_text:
                logger.info(
                    "Session %s agent reply: \"%s\"",
                    self._session_id,
                    agent_text[:100],
                )
                await self.push_frame(TTSSpeakFrame(text=agent_text))

            if ended:
                logger.info(
                    "Session %s call ended — tearing down pipeline",
                    self._session_id,
                )
                await self.push_frame(EndFrame())
        finally:
            self._processing = False
