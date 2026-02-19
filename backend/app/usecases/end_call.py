"""
End-call use case.

Finalizes a session: builds the structured outcome from the current
ProspectState and DecisionTrace, then persists it to the session store.

Idempotent: if the session is already completed, returns the stored outcome.
"""

from __future__ import annotations

from app.domain.outcome import build_outcome
from app.domain.state import DecisionTrace, ProspectState
from app.infra.session_store import get_session, set_outcome


def end_call(
    session_id: str,
    state: ProspectState,
    trace: DecisionTrace,
    ended_reason: str = "simulation_complete",
) -> dict:
    """
    Build the outcome and mark the session as completed.

    Returns the outcome dict (matches OutcomeSchema shape).
    Idempotent: if already completed, returns stored outcome.
    """
    # Idempotency: don't re-finalize a completed session
    session = get_session(session_id)
    if session and session["status"] == "completed" and session["outcome"]:
        return session["outcome"]

    trace.ended_reason = trace.ended_reason or ended_reason
    outcome = build_outcome(state, trace)
    set_outcome(session_id, outcome, ended_reason)
    return outcome
