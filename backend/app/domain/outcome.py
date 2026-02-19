"""
Outcome builder — assembles the final simulation result.

Takes the terminal ProspectState and the full DecisionTrace and produces
a plain dict suitable for JSON serialization.  The Pydantic schema layer
(schemas/outcome.py) validates this at the HTTP boundary.
"""

from __future__ import annotations

from app.domain.qualification import (
    label_from_score,
    score_opportunity,
    score_opportunity_with_breakdown,
)
from app.domain.state import (
    DecisionTrace,
    ExtractedSignals,
    ProspectState,
)


def build_outcome(state: ProspectState, trace: DecisionTrace) -> dict:
    """
    Assemble the final structured outcome from the call.

    Returns a dict with:
      learned_fields      — what we discovered about the prospect
      opportunity_score    — 0-10 numeric rating
      opportunity_label    — "Weak" / "Medium" / "Strong"
      recommended_next_action — what the sales team should do
      summary              — one-paragraph narrative
      decision_trace       — condensed list of per-turn decisions
    """
    # Compute final score with breakdown
    final_score, breakdown, explanation = score_opportunity_with_breakdown(
        state, ExtractedSignals()
    )
    label = label_from_score(final_score)
    next_action = _recommended_next_action(state, final_score, label)
    summary = _build_summary(state, final_score, label, next_action, trace)

    return {
        "learned_fields": dict(state.learned_fields),
        "opportunity_score": round(final_score, 1),
        "opportunity_label": label,
        "recommended_next_action": next_action,
        "summary": summary,
        "score_breakdown": breakdown,
        "score_explanation": explanation,
        "decision_trace": [t.to_dict() for t in trace.turns],
    }


def _recommended_next_action(
    state: ProspectState, score: float, label: str
) -> str:
    """Pick a concrete next-action recommendation based on score and state."""
    filled = len(state.filled_slots())

    if filled == 0:
        return "Re-attempt call with revised opener"

    if label == "Strong":
        if state.learned_fields.get("authority") is True:
            return "Fast-track to demo with decision-maker"
        return "Schedule discovery call with AE to reach decision-maker"

    if label == "Medium":
        return "Schedule discovery call with AE"

    # Weak
    if "not_interested" in state.objections:
        return "Move to nurture track; re-engage in 60 days"
    return "Nurture via email drip; re-engage in 30 days"


def _build_summary(
    state: ProspectState,
    score: float,
    label: str,
    next_action: str,
    trace: DecisionTrace,
) -> str:
    """Generate a one-paragraph human-readable summary of the call."""
    filled = len(state.filled_slots())
    total = 5
    turns = len(trace.turns)
    ended = trace.ended_reason or "agent decision"

    parts = [
        f"Cold-call simulation completed in {turns} turns (ended: {ended}).",
        f"Gathered {filled}/{total} qualification fields.",
    ]

    if state.objections:
        parts.append(f"Objections encountered: {', '.join(state.objections)}.")

    parts.append(f"Opportunity rated '{label}' ({score:.1f}/10).")
    parts.append(f"Recommendation: {next_action}.")

    return " ".join(parts)
