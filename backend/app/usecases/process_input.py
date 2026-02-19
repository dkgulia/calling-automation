"""
Process-input use case (Phase 4+5: LLM extraction + wording + hardening).

Handles a single user turn in an interactive session:
  1.  Load session state + trace
  2.  Try LLM extraction → fall back to rule-based if fails or low confidence
  3.  Update state with extracted signals (confidence-gated)
  4.  Recompute opportunity score
  5.  Run decision engine to pick next action
  6.  Transition call stage
  7.  Try LLM wording (using *after-update* state snapshot) → fall back to templates
  8.  Record everything in the trace (with source tracking)
  9.  Always persist state + trace (ensures final turn is saved even on end_call error)
  10. If action is END or CLOSE, finalize the session via end_call

The function is async so it can call DeepSeek without blocking the
Pipecat voice pipeline or the FastAPI event loop.
"""

from __future__ import annotations

import time

from app.core.logging import logger
from app.core.settings import settings
from app.domain.decision_engine import (
    decide_next_action,
    next_stage_from_action,
)
from app.domain.qualification import score_opportunity
from app.domain.state import (
    Action,
    CallStage,
    DecisionTrace,
    ExtractedSignals,
    ProspectState,
    extract_signals_rule_based,
)
from app.infra.providers.deepseek_r1 import (
    extract_signals_llm,
    generate_agent_utterance_llm,
)
from app.infra.session_store import get_session, save_session
from app.usecases.end_call import end_call


async def process_input(session_id: str, user_text: str) -> dict:
    """
    Process one user utterance and return the agent's response.

    Returns dict with status, agent_text, opportunity_score, ended, outcome.
    Uses LLM extraction + wording with automatic fallback to rule-based.
    """
    session = get_session(session_id)
    if session is None:
        return {"error": "session_not_found"}

    # If already completed, return the stored outcome
    if session["status"] == "completed":
        return {
            "status": "completed",
            "agent_text": None,
            "opportunity_score": session["outcome"]["opportunity_score"],
            "ended": True,
            "outcome": session["outcome"],
        }

    state: ProspectState = session["state"]
    trace: DecisionTrace = session["trace"]

    state.turn_count += 1
    state.last_user_text = user_text

    # --- 1. Extract signals (LLM with fallback) ---
    extracted_source = "rule_based"
    state_snapshot_before = _state_snapshot(state)
    use_llm = not settings.force_rule_based

    t_extract = time.monotonic()
    if use_llm:
        try:
            signals = await extract_signals_llm(
                session_id, user_text, state_snapshot_before
            )
            if signals.confidence < settings.llm_min_confidence:
                logger.info(
                    "Session %s: LLM confidence %.2f < %.2f, falling back to rule-based",
                    session_id,
                    signals.confidence,
                    settings.llm_min_confidence,
                )
                signals = extract_signals_rule_based(user_text)
            else:
                extracted_source = "llm"
        except Exception as e:
            logger.warning(
                "Session %s: LLM extraction failed (%s), using rule-based",
                session_id,
                e,
            )
            signals = extract_signals_rule_based(user_text)
    else:
        signals = extract_signals_rule_based(user_text)
    extract_ms = (time.monotonic() - t_extract) * 1000

    # --- 2. Snapshot score before update ---
    score_before = state.interest_score
    objections_before = set(state.objections)  # capture BEFORE update

    # --- 3. Merge signals (confidence-gated) ---
    state.update_from_signals(signals, min_confidence=settings.llm_min_confidence)

    # --- 4. Recompute score ---
    score_after = score_opportunity(state, signals)
    state.interest_score = score_after

    # --- 5. Decide next action ---
    action = decide_next_action(state, signals, prior_objections=objections_before)

    # --- 6. Transition stage ---
    stage_before = state.stage
    stage_after = next_stage_from_action(state.stage, action)
    state.stage = stage_after

    # After-update snapshot reflects newly learned slots + new stage
    state_snapshot_after = _state_snapshot(state)

    # --- 7. Generate agent text (LLM with fallback) ---
    # Uses state_snapshot_after so wording sees the latest context.
    wording_source = "template"
    t_wording = time.monotonic()
    if use_llm:
        try:
            agent_text = await generate_agent_utterance_llm(
                action, state_snapshot_after, signals.to_dict()
            )
            wording_source = "llm"
        except Exception as e:
            logger.warning(
                "Session %s: LLM wording failed (%s), using template",
                session_id,
                e,
            )
            agent_text = _generate_agent_text(action, state, signals)
    else:
        agent_text = _generate_agent_text(action, state, signals)
    wording_ms = (time.monotonic() - t_wording) * 1000

    state.last_agent_text = agent_text

    # --- 8. Record trace ---
    trace.add_turn(
        turn_index=state.turn_count,
        user_text=user_text,
        agent_text=agent_text,
        extracted=signals,
        action=action,
        score_before=score_before,
        score_after=score_after,
        stage_before=stage_before,
        stage_after=stage_after,
        extracted_source=extracted_source,
        wording_source=wording_source,
    )

    logger.info(
        "Session %s turn %d: %s -> %s (score %.1f -> %.1f, stage %s -> %s) "
        "[%s/%s] extract=%.0fms wording=%.0fms",
        session_id,
        state.turn_count,
        signals.intent,
        action.type,
        score_before,
        score_after,
        stage_before.value,
        stage_after.value,
        extracted_source,
        wording_source,
        extract_ms,
        wording_ms,
    )

    # --- 9. Always persist state + trace ---
    # This ensures the final turn is saved even if end_call errors later.
    save_session(session_id, state, trace)

    # --- 10. Finalize if the call is over ---
    ended = action.type in ("END", "CLOSE")
    outcome = None
    if ended:
        reason = action.reason_codes[0] if action.reason_codes else "agent_decision"
        trace.ended_reason = reason
        outcome = end_call(session_id, state, trace, ended_reason=reason)
        logger.info(
            "Session %s ended: %s (score %.1f, %s)",
            session_id,
            reason,
            outcome["opportunity_score"],
            outcome["opportunity_label"],
        )

    return {
        "status": "completed" if ended else "running",
        "agent_text": agent_text,
        "opportunity_score": round(score_after, 1),
        "ended": ended,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_snapshot(state: ProspectState) -> dict:
    """Create a plain dict snapshot of state for LLM context."""
    return {
        "stage": state.stage.value,
        "learned_fields": dict(state.learned_fields),
        "objections": list(state.objections),
        "turn_count": state.turn_count,
        "last_agent_text": state.last_agent_text,
        "last_user_text": state.last_user_text,
    }


# ---------------------------------------------------------------------------
# Deterministic agent text (template fallback — preserved from Phase 2)
# ---------------------------------------------------------------------------

_SLOT_QUESTIONS: dict[str, str] = {
    "pain": (
        "I'd love to understand your current workflow better. "
        "What's the biggest challenge your team faces with outbound right now?"
    ),
    "company_size": (
        "That's helpful context. Roughly how large is your team — "
        "how many people are involved in outbound?"
    ),
    "authority": (
        "Got it. And when it comes to evaluating a new tool like this, "
        "are you the one who makes that call, or is there someone else involved?"
    ),
    "budget": (
        "Makes sense. Is there budget set aside for improving your outbound "
        "process, or is that something you'd need to get approved?"
    ),
    "timeline": (
        "Understood. If this were a good fit, what would your timeline look "
        "like for getting something like this up and running?"
    ),
}

_OBJECTION_RESPONSES: dict[str, str] = {
    "not_interested": (
        "I totally understand — I wouldn't want to waste your time. "
        "Just out of curiosity, how is your team currently handling outbound? "
        "I ask because a lot of teams in your space have told us about similar frustrations."
    ),
    "already_have_tool": (
        "That's great that you already have something in place! "
        "A lot of our customers actually came from other tools. "
        "What would you say is the one thing you wish worked better with your current setup?"
    ),
    "too_expensive": (
        "I hear you on cost — it's always a factor. Most of our customers "
        "find the ROI pays for itself within the first quarter. "
        "What does your current cost per qualified meeting look like?"
    ),
    "send_email": (
        "Absolutely, I'll send that over right after this. Before I do — "
        "just so I can tailor the info — what's the biggest pain point "
        "you'd want a solution to address?"
    ),
    "busy": (
        "Totally respect that. I can absolutely call you back — "
        "would the time you mentioned work, or is there a better slot?"
    ),
}

_CLOSE_TEXT = (
    "Based on everything you've shared, it sounds like there could be a really "
    "strong fit here. Would you be open to a 30-minute demo this week so I can "
    "show you exactly how this would work for your team?"
)

_END_TEXTS: dict[str, str] = {
    "USER_ENDED": (
        "Totally understand. Thanks for taking the time to chat — "
        "I really appreciate it. Have a great rest of your day!"
    ),
    "TURN_LIMIT_REACHED": (
        "I know I've taken a lot of your time, so I'll let you go. "
        "Thanks for the conversation — I'll follow up with a summary. Have a great day!"
    ),
    "ALL_SLOTS_FILLED": (
        "I appreciate you sharing all of that. It sounds like the timing "
        "might not be right just now, but I'd love to stay in touch. "
        "I'll send over some info — thanks for your time!"
    ),
    "SCORE_TOO_LOW": (
        "I appreciate your honesty. It sounds like this might not be the best "
        "fit right now, and that's totally okay. I'll send some info in case "
        "anything changes down the road. Thanks for your time!"
    ),
    "SILENCE_TIMEOUT": (
        "It seems like we've lost you — no worries at all! "
        "I'll follow up with an email. Thanks for your time!"
    ),
}

_END_DEFAULT = (
    "Thanks so much for the conversation. I'll follow up with a summary "
    "and next steps. Have a great day!"
)


def _generate_agent_text(
    action: Action,
    state: ProspectState,
    signals: ExtractedSignals,
) -> str:
    """
    Generate a deterministic agent reply based on the chosen action.
    Used as fallback when LLM wording fails.
    """
    if action.type == "ASK_SLOT" and action.slot:
        return _SLOT_QUESTIONS.get(
            action.slot, f"Can you tell me more about your {action.slot}?"
        )

    if action.type == "HANDLE_OBJECTION":
        obj_type = signals.objection_type or "unknown"
        return _OBJECTION_RESPONSES.get(
            obj_type,
            "I understand your concern. Could you help me understand "
            "a bit more about what's holding you back?",
        )

    if action.type == "CLOSE":
        return _CLOSE_TEXT

    if action.type == "END":
        for reason in action.reason_codes:
            if reason in _END_TEXTS:
                return _END_TEXTS[reason]
        return _END_DEFAULT

    return _END_DEFAULT
