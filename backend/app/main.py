import asyncio
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import router as v1_router
from app.core.logging import logger, setup_logging
from app.core.settings import settings
from app.infra.session_store import get_active_session_ids, get_session

setup_logging()

app = FastAPI(
    title="Roister Cold-Call Simulation",
    version="0.3.0",
    description="Conversational cold-call simulation API with Pipecat voice + LLM",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)


@app.get("/health")
async def health():
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Pipecat real-time voice endpoint.

    Connect with: ws://localhost:8000/ws?session_id=<id>
    The session must already exist (created via POST /api/v1/run).
    """
    from app.infra.pipecat.pipeline import run_pipeline

    session = get_session(session_id)
    if session is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    if session["status"] != "running":
        await websocket.close(code=4010, reason="Session already completed")
        return

    logger.info("WebSocket connection for session %s", session_id)

    # Accept the WebSocket BEFORE passing to pipecat.
    # FastAPIWebsocketTransport fires on_client_connected before
    # iter_bytes() (which would auto-accept), so we must accept
    # explicitly to allow send_bytes() in the on_connected handler.
    await websocket.accept()

    # Debug: monitor raw WebSocket messages in background
    import asyncio as _asyncio

    original_receive = websocket.receive

    _msg_count = 0

    async def _logging_receive():
        nonlocal _msg_count
        msg = await original_receive()
        _msg_count += 1
        msg_type = msg.get("type", "?")
        has_bytes = len(msg.get("bytes", b"")) if "bytes" in msg else 0
        has_text = len(msg.get("text", "")) if "text" in msg else 0
        if _msg_count <= 5 or _msg_count % 200 == 0:
            logger.info(
                "Session %s: raw WS msg #%d type=%s bytes=%d text=%d",
                session_id, _msg_count, msg_type, has_bytes, has_text,
            )
        return msg

    websocket.receive = _logging_receive  # type: ignore

    try:
        await run_pipeline(websocket, session_id)
    except WebSocketDisconnect:
        logger.info("Session %s: WebSocket disconnected by client", session_id)
    except Exception as e:
        logger.error("Session %s: pipeline error: %s", session_id, e, exc_info=True)


# ---------------------------------------------------------------------------
# Silence timeout watchdog (Phase 5)
# ---------------------------------------------------------------------------

async def _silence_watchdog() -> None:
    """Periodically check for sessions that have been silent too long."""
    from app.usecases.end_call import end_call

    timeout = settings.silence_timeout_seconds
    while True:
        await asyncio.sleep(2)
        now = time.monotonic()

        for sid in get_active_session_ids():
            session = get_session(sid)
            if session is None or session["status"] != "running":
                continue

            last_at = session.get("last_user_at", now)
            elapsed = now - last_at
            if elapsed > timeout:
                logger.info(
                    "Session %s: silence timeout (%.0fs > %ds)",
                    sid,
                    elapsed,
                    timeout,
                )
                state = session["state"]
                trace = session["trace"]
                trace.ended_reason = "SILENCE_TIMEOUT"
                end_call(sid, state, trace, ended_reason="SILENCE_TIMEOUT")


@app.on_event("startup")
async def _start_watchdog():
    asyncio.create_task(_silence_watchdog())
