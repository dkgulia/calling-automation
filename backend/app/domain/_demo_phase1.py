"""
Phase 1 self-check demo.

Run from the backend/ directory:
    python -m app.domain._demo_phase1

Walks through 5 example user texts, running each through the full
domain pipeline: extract -> update state -> score -> decide -> trace.
Prints the final outcome JSON at the end.
"""

from __future__ import annotations

import json

from app.domain.decision_engine import (
    agent_goal_for_action,
    decide_next_action,
    next_stage_from_action,
)
from app.domain.outcome import build_outcome
from app.domain.qualification import score_opportunity
from app.domain.state import (
    CallStage,
    DecisionTrace,
    ProspectState,
    extract_signals_rule_based,
)


def main() -> None:
    user_texts = [
        "Hi, what's this about?",
        "We already use a tool for this.",
        "We are 50 people, marketing team is 4.",
        "Budget depends, maybe later.",
        "Ok, send details.",
    ]

    state = ProspectState(session_id="demo_001")
    trace = DecisionTrace(session_id="demo_001")

    print("=" * 70)
    print("PHASE 1 DOMAIN DEMO — Cold-Call Simulation Engine")
    print("=" * 70)

    for i, user_text in enumerate(user_texts):
        turn = i + 1
        state.turn_count = turn
        state.last_user_text = user_text

        # 1. Extract signals
        signals = extract_signals_rule_based(user_text)

        # 2. Snapshot score before update
        score_before = state.interest_score

        # 3. Update state from signals
        state.update_from_signals(signals)

        # 4. Recompute score
        score_after = score_opportunity(state, signals)
        state.interest_score = score_after

        # 5. Decide next action
        action = decide_next_action(state, signals)

        # 6. Transition stage
        stage_before = state.stage
        stage_after = next_stage_from_action(state.stage, action)
        state.stage = stage_after

        # 7. Generate agent goal text
        agent_text = agent_goal_for_action(action, state)
        state.last_agent_text = agent_text

        # 8. Record trace
        trace.add_turn(
            turn_index=turn,
            user_text=user_text,
            agent_text=agent_text,
            extracted=signals,
            action=action,
            score_before=score_before,
            score_after=score_after,
            stage_before=stage_before,
            stage_after=stage_after,
        )

        # Print turn summary
        print(f"\n--- Turn {turn} ---")
        print(f"  User:       \"{user_text}\"")
        print(f"  Intent:     {signals.intent}"
              + (f" ({signals.objection_type})" if signals.objection_type else ""))
        print(f"  Signals:    pain={signals.pain}  size={signals.company_size}  "
              f"budget={signals.budget}  auth={signals.authority}  "
              f"timeline={signals.timeline}")
        print(f"  Score:      {score_before:.1f} -> {score_after:.1f}")
        print(f"  Stage:      {stage_before.value} -> {stage_after.value}")
        print(f"  Action:     {action.type}"
              + (f" (slot={action.slot})" if action.slot else ""))
        print(f"  Goal:       {agent_text}")
        print(f"  Reasons:    {action.reason_codes}")

        # Stop if the engine says END
        if action.type == "END" or state.stage == CallStage.END:
            trace.ended_reason = (
                action.reason_codes[0] if action.reason_codes else "agent_decision"
            )
            break

    # Build final outcome
    trace.ended_reason = trace.ended_reason or "all_turns_processed"
    outcome = build_outcome(state, trace)

    print("\n" + "=" * 70)
    print("FINAL OUTCOME")
    print("=" * 70)
    print(json.dumps(outcome, indent=2, default=str))


if __name__ == "__main__":
    main()
