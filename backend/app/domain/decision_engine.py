"""
Decision engine — the "brain" that chooses the agent's next move.

Core principle: **no linear script**.  The engine inspects the current
ProspectState (filled slots, objections, stage, score) and the latest
ExtractedSignals to pick the single best action.  This means the
conversation can recover from detours, handle objections mid-qualify,
and close early when signals are strong.

The engine is a pure function (state + signals -> action) with no side
effects, making it trivial to unit-test and replay.
"""

from __future__ import annotations

from app.domain.state import (
    Action,
    CallStage,
    ExtractedSignals,
    ProspectState,
    SLOT_PRIORITY,
)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def decide_next_action(
    state: ProspectState,
    signals: ExtractedSignals,
    prior_objections: set[str] | None = None,
) -> Action:
    """
    Given the current state and freshly-extracted signals, decide what the
    agent should do next.

    Decision priority (highest to lowest):
      1. Explicit end signal or turn limit  -> END
      2. Weak-lead bail-out                 -> END
      3. Objection detected (new only)      -> HANDLE_OBJECTION (max 2 total)
      4. All key slots filled + strong lead -> CLOSE
      5. Missing slots remain               -> ASK_SLOT (best next slot)
      6. Fallback                           -> END (nothing left to do)

    Args:
        prior_objections: objections already in state BEFORE this turn's update.
            Used to determine if the current objection is new or already handled.

    Returns an Action with type, optional slot, message_goal, and reason_codes
    that explain *why* this action was chosen.
    """
    reasons: list[str] = []
    _prior = prior_objections if prior_objections is not None else set()

    # --- 1. Hard stop: user wants to end, or we've hit the turn limit ---
    if signals.intent == "end":
        reasons.append("USER_ENDED")
        return Action(
            type="END",
            message_goal="Wrap up politely — prospect signaled end of call",
            reason_codes=reasons,
        )

    if state.turn_count >= 10:
        reasons.append("TURN_LIMIT_REACHED")
        return Action(
            type="END",
            message_goal="Wrap up — reached maximum turns for this call",
            reason_codes=reasons,
        )

    # --- 2. Weak-lead bail-out: all slots filled but score too low ---
    missing = state.missing_slots()
    if not missing and state.interest_score < 4:
        reasons.append("ALL_SLOTS_FILLED")
        reasons.append("SCORE_TOO_LOW")
        return Action(
            type="END",
            message_goal="Thank prospect and end — lead is too weak to pursue",
            reason_codes=reasons,
        )

    # --- 3. Handle objection (only if NEW and we haven't handled too many) ---
    if signals.intent == "objection" and signals.objection_type:
        # "Busy" / callback requests → always respect and end call gracefully
        if signals.objection_type == "busy":
            reasons.append("CALLBACK_REQUESTED")
            return Action(
                type="END",
                message_goal="Agree to call back at the time they requested and say a friendly goodbye",
                reason_codes=reasons,
            )
        is_new = signals.objection_type not in _prior
        total_handled = len(_prior)  # how many objections handled before this turn
        if is_new and total_handled < 2:
            reasons.append(f"OBJECTION_{signals.objection_type.upper()}")
            return Action(
                type="HANDLE_OBJECTION",
                message_goal=_objection_goal(signals.objection_type),
                reason_codes=reasons,
            )
        # Already handled or hit objection cap — fall through to slot probing
        reasons.append(f"OBJECTION_SKIPPED_{signals.objection_type.upper()}")

    # --- 4. All key slots filled + strong enough score -> close ---
    if not missing and state.interest_score >= 6:
        reasons.append("ALL_SLOTS_FILLED")
        reasons.append("SCORE_STRONG_ENOUGH")
        return Action(
            type="CLOSE",
            message_goal="Propose next step — all qualifications met",
            reason_codes=reasons,
        )

    # --- 5. Probe next missing slot ---
    if missing:
        # In INTRO stage, always start with pain to transition to DISCOVERY
        if state.stage == CallStage.INTRO:
            next_slot = "pain"
            reasons.append("INTRO_TO_DISCOVERY")
        else:
            next_slot = _pick_best_slot(missing, state)

        reasons.append(f"MISSING_SLOT_{next_slot.upper()}")
        return Action(
            type="ASK_SLOT",
            slot=next_slot,
            message_goal=_slot_goal(next_slot),
            reason_codes=reasons,
        )

    # --- 6. Fallback ---
    reasons.append("FALLBACK_NO_ACTION")
    return Action(
        type="END",
        message_goal="No further actions available — end call gracefully",
        reason_codes=reasons,
    )


# ---------------------------------------------------------------------------
# Stage transitions
# ---------------------------------------------------------------------------

def next_stage_from_action(current: CallStage, action: Action) -> CallStage:
    """
    Determine the next CallStage based on the chosen action.

    Stage machine:
      ASK_SLOT          -> DISCOVERY (from INTRO) or QUALIFY
      HANDLE_OBJECTION  -> OBJECTION
      CLOSE             -> CLOSE
      END               -> END
    """
    if action.type == "END":
        return CallStage.END

    if action.type == "CLOSE":
        return CallStage.CLOSE

    if action.type == "HANDLE_OBJECTION":
        return CallStage.OBJECTION

    if action.type == "ASK_SLOT":
        if current == CallStage.INTRO:
            return CallStage.DISCOVERY
        # After handling an objection, go back to qualifying
        if current == CallStage.OBJECTION:
            return CallStage.QUALIFY
        return CallStage.QUALIFY

    return current  # no change


# ---------------------------------------------------------------------------
# Human-readable goal text
# ---------------------------------------------------------------------------

def agent_goal_for_action(action: Action, state: ProspectState) -> str:
    """
    Produce a short, human-readable goal string describing what the agent
    should accomplish in its next utterance.  This feeds into the LLM system
    prompt in later phases.
    """
    if action.type == "END":
        return action.message_goal

    if action.type == "CLOSE":
        filled = len(state.filled_slots())
        return (
            f"All {filled} qualification slots filled (score {state.interest_score:.1f}). "
            f"Propose a concrete next step: demo, meeting, or trial."
        )

    if action.type == "HANDLE_OBJECTION":
        return action.message_goal

    if action.type == "ASK_SLOT" and action.slot:
        return f"Probe for '{action.slot}': {action.message_goal}"

    return action.message_goal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_best_slot(missing: list[str], state: ProspectState) -> str:
    """
    Choose the next slot to probe from the missing list.

    Uses SLOT_PRIORITY order but skips slots that are unlikely to yield
    answers right now (e.g., don't ask budget before establishing pain).
    """
    # Default: follow priority order
    for slot in SLOT_PRIORITY:
        if slot in missing:
            return slot
    # Shouldn't happen, but just in case
    return missing[0]


_SLOT_GOALS: dict[str, str] = {
    "pain": "Ask about current challenges and pain points in their workflow",
    "company_size": "Ask about team or company size to understand scale",
    "authority": "Determine if the prospect is the decision-maker",
    "budget": "Explore whether budget is available for a solution",
    "timeline": "Understand their timeline for implementing a solution",
}


def _slot_goal(slot: str) -> str:
    return _SLOT_GOALS.get(slot, f"Probe for information about {slot}")


_OBJECTION_GOALS: dict[str, str] = {
    "not_interested": "Acknowledge disinterest, ask one clarifying question about their current approach",
    "already_have_tool": "Acknowledge their current tool, explore gaps or frustrations with it",
    "too_expensive": "Reframe around ROI and value, ask what budget range would work",
    "send_email": "Agree to send info, but first ask one qualifying question",
    "busy": "Respect their time, offer to schedule a brief 5-min follow-up",
}


def _objection_goal(objection_type: str) -> str:
    return _OBJECTION_GOALS.get(
        objection_type,
        f"Address the '{objection_type}' objection with empathy and a pivot question",
    )
