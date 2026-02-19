"""
Generate-prospect-turn use case (Phase 6: AI Prospect mode).

When the session is in "ai" prospect mode, this generates the next
prospect utterance (either via LLM or scripted fallback), then feeds it
through process_input() exactly as if a human had typed it.
"""

from __future__ import annotations

from app.core.logging import logger
from app.infra.providers.deepseek_r1 import generate_prospect_utterance_llm
from app.infra.session_store import get_session
from app.usecases.process_input import process_input

# Scripted fallback responses by turn number
_SCRIPTED_PROSPECT = [
    "Yeah sure, I have a minute. What's this about?",
    "We're about 50 people. Outbound is mostly manual right now, lots of spreadsheets.",
    "I handle the sales tools decisions, yeah.",
    "Honestly, we've looked at a few things but nothing stuck. Budget is there if it makes sense.",
    "Probably this quarter if the fit is right.",
    "Sure, I'd be open to a demo.",
]


async def generate_prospect_turn(session_id: str) -> dict:
    """
    Generate an AI prospect utterance and process it as a turn.

    Returns the same dict shape as process_input().
    """
    session = get_session(session_id)
    if session is None:
        return {"error": "session_not_found"}

    if session["status"] == "completed":
        return {
            "status": "completed",
            "agent_text": None,
            "prospect_text": None,
            "ended": True,
            "outcome": session["outcome"],
        }

    state = session["state"]
    agent_text = state.last_agent_text or ""
    turn = state.turn_count

    # Build state snapshot for LLM context
    state_snapshot = {
        "stage": state.stage.value,
        "learned_fields": dict(state.learned_fields),
        "turn_count": turn,
        "objections": list(state.objections),
    }

    # Try LLM prospect generation, fall back to scripted
    try:
        prospect_text = await generate_prospect_utterance_llm(
            agent_text, state_snapshot, session_id=session_id
        )
        logger.info(
            "Session %s: AI prospect (llm): \"%s\"",
            session_id,
            prospect_text[:100],
        )
    except Exception as e:
        logger.warning(
            "Session %s: AI prospect LLM failed (%s), using scripted",
            session_id,
            e,
        )
        idx = min(turn, len(_SCRIPTED_PROSPECT) - 1)
        prospect_text = _SCRIPTED_PROSPECT[idx]

    # Feed through the normal turn pipeline
    result = await process_input(session_id, prospect_text)
    result["prospect_text"] = prospect_text
    return result
