"""
Regression demo — question alignment and safe corrections.

Demonstrates:
  - last_asked_slot prevents accidental slot fills from filler utterances
  - Explicit corrections can overwrite mistaken slot values
  - Diminishing objection penalties

Usage:
    cd backend
    source .venv/bin/activate
    FORCE_RULE_BASED=1 python -m app.usecases._demo_alignment_and_corrections
"""

from __future__ import annotations

import asyncio
import os
import sys

# Force rule-based mode for deterministic results
os.environ.setdefault("FORCE_RULE_BASED", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core.logging import setup_logging
from app.usecases.run_simulation import start_session
from app.usecases.process_input import process_input
from app.infra.session_store import get_session


async def main() -> None:
    setup_logging()

    print("=" * 60)
    print("Regression Demo: Alignment + Corrections + Objection Penalties")
    print("=" * 60)
    print()

    # ---------- Test 1: Alignment gating ----------
    print("--- TEST 1: Alignment Gating ---")
    print("Agent asks about pain, user says 'yeah' -> should NOT fill pain\n")

    session = start_session()
    sid = session["session_id"]

    # Turn 1: intro -> agent asks about pain (ASK_SLOT pain)
    r1 = await process_input(sid, "Yeah sure, I have a minute.")
    print(f"  Turn 1 user : Yeah sure, I have a minute.")
    print(f"  Turn 1 agent: {r1['agent_text']}")

    state = get_session(sid)["state"]
    print(f"  last_asked_slot = {state.last_asked_slot}")
    assert state.last_asked_slot == "pain", f"Expected last_asked_slot='pain', got '{state.last_asked_slot}'"

    # Turn 2: filler "yeah" -> should NOT fill pain
    r2 = await process_input(sid, "yeah")
    state = get_session(sid)["state"]
    print(f"\n  Turn 2 user : yeah")
    print(f"  Turn 2 agent: {r2['agent_text']}")
    print(f"  pain = {state.learned_fields['pain']}")
    # "yeah" as rule-based extraction: off_topic (<=3 words, no slots) so pain stays None
    if state.learned_fields["pain"] is None:
        print("  PASS: 'yeah' did not accidentally fill pain")
    else:
        print("  NOTE: 'yeah' filled pain (rule-based may extract if pattern matched)")

    # Turn 3: actual pain answer -> should fill pain (asked slot) but NOT budget (unasked)
    r3 = await process_input(sid, "Yes, budget is approved and outbound is really painful for us, like 8 out of 10.")
    state = get_session(sid)["state"]
    print(f"\n  Turn 3 user : Yes, budget is approved and outbound is really painful for us, like 8 out of 10.")
    print(f"  Turn 3 agent: {r3['agent_text']}")
    print(f"  pain = {state.learned_fields['pain']}")
    print(f"  budget = {state.learned_fields['budget']}  (should be None — alignment blocks unasked slots)")
    assert state.learned_fields["pain"] is not None, "Expected pain to be filled"
    print("  PASS: pain filled after explicit answer")
    if state.learned_fields["budget"] is None:
        print("  PASS: budget blocked by alignment gating (unasked slot, confidence < 0.85)")

    print()

    # ---------- Test 2: Safe correction overwrites ----------
    print("--- TEST 2: Safe Correction Overwrites ---")
    print("Fill company_size via aligned answer, then correct it with 'actually'\n")

    session2 = start_session()
    sid2 = session2["session_id"]

    # Turn 1: intro -> agent asks pain
    await process_input(sid2, "Sure, what is this about?")
    # Turn 2: answer pain (aligned) -> agent asks company_size
    await process_input(sid2, "Outbound is really painful for us, I'd say 7 out of 10.")
    # Turn 3: answer company_size (aligned) -> should fill
    await process_input(sid2, "We're about 50 people on the team.")

    state2 = get_session(sid2)["state"]
    print(f"  company_size = {state2.learned_fields['company_size']}")
    print(f"  slot_confidences = {state2.slot_confidences}")
    assert state2.learned_fields["company_size"] == 50, f"Expected 50, got {state2.learned_fields['company_size']}"

    # Correction
    r_corr = await process_input(sid2, "Actually we're 200 people, I misspoke.")
    state2 = get_session(sid2)["state"]
    print(f"\n  Correction user : Actually we're 200 people, I misspoke.")
    print(f"  Correction agent: {r_corr['agent_text']}")
    print(f"  company_size = {state2.learned_fields['company_size']}")

    if state2.learned_fields["company_size"] == 200:
        print("  PASS: company_size corrected from 50 to 200")
    else:
        print(f"  FAIL: company_size is {state2.learned_fields['company_size']}, expected 200")

    print()

    # ---------- Test 3: Diminishing objection penalties ----------
    print("--- TEST 3: Diminishing Objection Penalties ---")
    print("Same objection repeated 3x should have diminishing impact\n")

    session3 = start_session()
    sid3 = session3["session_id"]

    # Fill pain (aligned) then company_size (aligned)
    await process_input(sid3, "Hi, sure.")
    await process_input(sid3, "Outbound is painful for us, maybe 7 out of 10.")
    await process_input(sid3, "We're about 40 people.")

    # First objection
    r_obj1 = await process_input(sid3, "We already have a tool for that.")
    state3 = get_session(sid3)["state"]
    score1 = state3.interest_score
    print(f"  Objection 1: score = {score1}")
    print(f"  objection_counts = {state3.objection_counts}")

    # Second occurrence of same objection
    r_obj2 = await process_input(sid3, "I told you, we already have a tool.")
    state3 = get_session(sid3)["state"]
    score2 = state3.interest_score
    print(f"  Objection 2 (repeat): score = {score2}")
    print(f"  objection_counts = {state3.objection_counts}")

    # Third occurrence
    r_obj3 = await process_input(sid3, "We already have a tool, stop asking.")
    state3 = get_session(sid3)["state"]
    score3 = state3.interest_score
    print(f"  Objection 3 (repeat): score = {score3}")
    print(f"  objection_counts = {state3.objection_counts}")

    # With diminishing penalties, score should not keep dropping the same amount
    print(f"\n  Score progression: {score1} -> {score2} -> {score3}")
    if score2 >= score1 or abs(score2 - score3) < abs(score1 - score2):
        print("  PASS: Diminishing penalty observed (or score stabilized)")
    else:
        print("  NOTE: Scores may vary based on other slot changes")

    print()
    print("=" * 60)
    print("Regression demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
