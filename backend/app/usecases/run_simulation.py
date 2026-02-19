"""
Run-simulation use case (Phase 2: interactive text mode).

Creates a new session with fresh ProspectState and DecisionTrace,
generates the opening agent message, and returns everything the
caller needs to start sending user input.

No background task — the caller drives the conversation turn-by-turn
via POST /api/v1/input/{session_id}.
"""

from __future__ import annotations

from app.core.logging import logger
from app.domain.state import DecisionTrace, ProspectState
from app.infra.session_store import create_session
from app.utils.ids import generate_session_id

# The opening line the agent uses to start the call.
# Public constant — also imported by infra/pipecat/pipeline.py for the voice opener.
OPENER = (
    "Hi there! This is Alex from Roister. I help teams streamline their "
    "outbound sales process. Do you have a quick moment to chat about how "
    "your team currently handles cold outreach?"
)


def start_session(prospect_mode: str = "human") -> dict:
    """
    Initialize a new interactive simulation session.

    Args:
        prospect_mode: "human" (mic) or "ai" (LLM-generated prospect turns).

    Returns:
        dict with session_id, status, agent_text, and connect_info.
    """
    session_id = generate_session_id()

    state = ProspectState(session_id=session_id)
    trace = DecisionTrace(session_id=session_id)

    # Store live state so /input can pick it up
    create_session(session_id, state, trace, prospect_mode=prospect_mode)

    logger.info(
        "Session created: %s (mode=%s)", session_id, prospect_mode
    )

    return {
        "session_id": session_id,
        "status": "running",
        "agent_text": OPENER,
        "prospect_mode": prospect_mode,
        "connect_info": {
            "ws_url": f"ws://localhost:8000/ws?session_id={session_id}",
        },
    }
