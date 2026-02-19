"""
In-memory session store.

Stores the full live session state (ProspectState + DecisionTrace) alongside
status and outcome.  This allows the /input endpoint to load, mutate, and
re-persist state on every turn without the caller needing to hold anything.

Tradeoff: in-memory dict means single-process only.  Swap for Redis or
Postgres in production.
"""

from __future__ import annotations

import time
from typing import Any

from app.domain.state import DecisionTrace, ProspectState

_store: dict[str, dict[str, Any]] = {}


def create_session(
    session_id: str,
    state: ProspectState,
    trace: DecisionTrace,
    prospect_mode: str = "human",
) -> None:
    """Create a new session with live state and trace."""
    _store[session_id] = {
        "state": state,
        "trace": trace,
        "status": "running",
        "outcome": None,
        "last_user_at": time.monotonic(),
        "prospect_mode": prospect_mode,
    }


def get_session(session_id: str) -> dict[str, Any] | None:
    return _store.get(session_id)


def save_session(session_id: str, state: ProspectState, trace: DecisionTrace) -> None:
    """Persist updated state and trace back to the store."""
    if session_id in _store:
        _store[session_id]["state"] = state
        _store[session_id]["trace"] = trace
        _store[session_id]["last_user_at"] = time.monotonic()


def set_outcome(session_id: str, outcome: dict, ended_reason: str = "simulation_complete") -> None:
    """Mark session completed and store the final outcome."""
    if session_id in _store:
        _store[session_id]["status"] = "completed"
        _store[session_id]["outcome"] = outcome


def get_active_session_ids() -> list[str]:
    """Return session IDs that are still running."""
    return [sid for sid, s in _store.items() if s["status"] == "running"]
