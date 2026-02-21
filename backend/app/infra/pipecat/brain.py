"""
BrainProcessor — bridges Pipecat's frame pipeline to the domain layer.

Sits between Deepgram STT (upstream) and Cartesia TTS (downstream) in the
Pipecat pipeline.  Uses VAD (Voice Activity Detection) signals to know when
the user has truly stopped speaking, then processes the accumulated text.

Flow:
  1. UserStartedSpeakingFrame arrives → mark user as speaking, cancel any
     pending processing timer
  2. TranscriptionFrames arrive → accumulate text fragments
  3. UserStoppedSpeakingFrame arrives → start a short debounce timer (1.0s)
     to catch any trailing transcriptions from Deepgram
  4. Timer fires → call process_input(session_id, text) with full utterance
  5. Push TTSSpeakFrame with agent_text downstream to TTS
  6. If call ended (action=END), wait for TTS to buffer, then push EndFrame

This is much more reliable than a blind debounce because VAD analyzes the
actual audio stream — it knows when the user pauses briefly (thinking) vs.
when they've actually finished their turn.
"""

from __future__ import annotations

import asyncio
import time

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.core.logging import logger
from app.usecases.process_input import process_input

# Safety margin after VAD says user stopped speaking.
# Catches trailing Deepgram transcriptions that arrive after speech ends.
_POST_SPEECH_DELAY = 1.0


class BrainProcessor(FrameProcessor):
    """Converts finalized speech transcriptions into agent voice responses."""

    def __init__(self, session_id: str, **kwargs):
        super().__init__(**kwargs)
        self._session_id = session_id
        self._processing = False       # guard against overlapping turns
        self._ended = False            # guard against post-end transcriptions
        self._user_speaking = False    # tracks VAD state
        self._pending_text: list[str] = []  # accumulated transcription fragments
        self._debounce_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # --- VAD: user started speaking ---
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            # Cancel any pending debounce — user is still talking
            if self._debounce_task and not self._debounce_task.done():
                self._debounce_task.cancel()
                logger.debug(
                    "Session %s: debounce cancelled — user started speaking",
                    self._session_id,
                )
            await self.push_frame(frame, direction)
            return

        # --- VAD: user stopped speaking ---
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            # Start debounce if we have text to process
            if (
                self._pending_text
                and not self._processing
                and not self._ended
            ):
                self._restart_debounce()
            await self.push_frame(frame, direction)
            return

        # Drop interim transcriptions — only final TranscriptionFrames matter.
        if isinstance(frame, InterimTranscriptionFrame):
            return

        if not isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)
            return

        text = frame.text.strip()
        if not text:
            return

        if self._ended:
            return

        # Buffer text while processing (picked up after current turn finishes)
        if self._processing:
            self._pending_text.append(text)
            logger.info(
                "Session %s: buffered transcription while processing: %s",
                self._session_id,
                text[:80],
            )
            return

        # Accumulate text
        self._pending_text.append(text)

        # Only start debounce if user has stopped speaking.
        # If still speaking, debounce will start when UserStoppedSpeakingFrame
        # arrives.  Fallback: if VAD never fires (e.g. text-only tests),
        # _user_speaking stays False and we debounce immediately.
        if not self._user_speaking:
            self._restart_debounce()

    def _restart_debounce(self) -> None:
        """Cancel any pending debounce and start a new one."""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounce_and_process())

    async def _debounce_and_process(self) -> None:
        """Wait for trailing transcriptions, then process the accumulated text."""
        try:
            await asyncio.sleep(_POST_SPEECH_DELAY)
        except asyncio.CancelledError:
            return  # user started speaking again, or new fragment arrived

        # Double-check user isn't speaking (race condition guard)
        if self._user_speaking:
            return

        if not self._pending_text:
            return
        combined = " ".join(self._pending_text)
        self._pending_text.clear()

        await self._handle_turn(combined)

    async def _handle_turn(self, text: str) -> None:
        """Process one complete user turn through the domain layer."""
        if self._ended or self._processing:
            if self._processing:
                self._pending_text.append(text)
            return

        self._processing = True
        try:
            logger.info("Session %s STT turn: \"%s\"", self._session_id, text)

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
                self._ended = True
                logger.info(
                    "Session %s call ended — waiting for TTS to buffer, then tearing down",
                    self._session_id,
                )
                await asyncio.sleep(0.5)
                await self.push_frame(EndFrame())
            else:
                # Check if more text accumulated while we were processing.
                # If so, process the next turn immediately.
                if self._pending_text:
                    combined = " ".join(self._pending_text)
                    self._pending_text.clear()
                    self._processing = False
                    await self._handle_turn(combined)
                    return

        finally:
            self._processing = False
