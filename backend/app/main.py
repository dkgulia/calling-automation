import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import router as v1_router
from app.core.logging import logger, setup_logging
from app.core.settings import settings
from app.infra.session_store import get_active_session_ids, get_session

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage background tasks across the app lifecycle."""
    watchdog_task = asyncio.create_task(_silence_watchdog())
    yield
    watchdog_task.cancel()
    try:
        await watchdog_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Roister Cold-Call Simulation",
    version="0.3.0",
    description="Conversational cold-call simulation API with Pipecat voice + LLM",
    lifespan=lifespan,
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

    try:
        await run_pipeline(websocket, session_id)
    except WebSocketDisconnect:
        logger.info("Session %s: WebSocket disconnected by client", session_id)
    except Exception as e:
        logger.error("Session %s: pipeline error: %s", session_id, e, exc_info=True)
    finally:
        # Ensure the WebSocket is closed after the pipeline finishes so the
        # frontend receives an onclose event and can fetch the outcome.
        try:
            await websocket.close(code=1000, reason="Pipeline finished")
        except Exception:
            pass
        logger.info("Session %s: WebSocket handler exiting", session_id)


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


