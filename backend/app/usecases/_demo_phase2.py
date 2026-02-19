"""
Phase 2 self-check demo — interactive text conversation loop.

Run from the backend/ directory:
    python -m app.usecases._demo_phase2

Starts a session, sends 6 sample user inputs through the full
process_input pipeline, and prints the agent reply + state at each
turn. Finishes by printing the complete outcome JSON.
"""

from __future__ import annotations

import json

from app.usecases.run_simulation import start_session
from app.usecases.process_input import process_input


def main() -> None:
    user_inputs = [
        "Hi, yeah I have a sec. What is this about?",
        "We're about 35 people. Our sales team does everything manually, it's really tedious.",
        "We already use HubSpot actually.",
        "Honestly the reporting is pretty slow and we waste a lot of time on data entry.",
        "I'm the director of sales, so yeah I make the call on tools.",
        "We've got budget set aside for this quarter. What would a demo look like?",
    ]

    print("=" * 70)
    print("PHASE 2 DEMO — Interactive Text Conversation")
    print("=" * 70)

    # 1. Start session
    session = start_session()
    session_id = session["session_id"]

    print(f"\n  Session: {session_id}")
    print(f"  Agent:   {session['agent_text']}")

    # 2. Send each user input
    for i, user_text in enumerate(user_inputs, 1):
        print(f"\n--- Turn {i} ---")
        print(f"  User:    \"{user_text}\"")

        result = process_input(session_id, user_text)

        print(f"  Agent:   {result['agent_text']}")
        print(f"  Score:   {result['opportunity_score']}")
        print(f"  Status:  {result['status']}")
        print(f"  Ended:   {result['ended']}")

        if result["ended"]:
            print("\n" + "=" * 70)
            print("CALL ENDED — FINAL OUTCOME")
            print("=" * 70)
            print(json.dumps(result["outcome"], indent=2, default=str))
            return

    # 3. If we got through all inputs without ending, fetch outcome
    print("\n(All inputs sent — session still running)")
    print("Sending final outcome request by ending the conversation...")

    result = process_input(session_id, "goodbye")
    print(f"\n  Agent:   {result['agent_text']}")
    if result["ended"] and result["outcome"]:
        print("\n" + "=" * 70)
        print("CALL ENDED — FINAL OUTCOME")
        print("=" * 70)
        print(json.dumps(result["outcome"], indent=2, default=str))


if __name__ == "__main__":
    main()
