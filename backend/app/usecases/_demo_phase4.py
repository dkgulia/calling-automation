"""
Phase 4 demo — runs a multi-turn conversation using async process_input.

Demonstrates:
  - LLM extraction + wording when DEEPSEEK_API_KEY is set
  - Automatic fallback to rule-based + templates when key is missing
  - Confidence gating and trace source tracking

Usage:
    cd backend
    source .venv/bin/activate
    python -m app.usecases._demo_phase4
"""

from __future__ import annotations

import asyncio
import json
import sys
import os

# Ensure backend is importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core.logging import setup_logging
from app.core.settings import settings
from app.usecases.run_simulation import start_session
from app.usecases.process_input import process_input

TURNS = [
    "Hi, what's this about?",
    "We're a team of 50 people and our outbound process is really manual and painful.",
    "I'm the VP of Sales, so yes I make these decisions.",
    "We have budget set aside for this quarter.",
    "We're looking to implement something by next quarter.",
    "That sounds interesting, let's set up a demo.",
]


async def main() -> None:
    setup_logging()

    print("=" * 60)
    print("Phase 4 Demo: LLM extraction + wording with fallback")
    print("=" * 60)

    if settings.deepseek_api_key:
        print(f"  DeepSeek API key : configured")
        print(f"  Model            : {settings.deepseek_model}")
        print(f"  Min confidence   : {settings.llm_min_confidence}")
    else:
        print("  DeepSeek API key : NOT SET")
        print("  Mode             : rule-based + template fallback")
    print()

    session = start_session()
    sid = session["session_id"]
    print(f"Session: {sid}")
    print(f"Agent  : {session['agent_text']}")
    print()

    for i, user_text in enumerate(TURNS, 1):
        print(f"--- Turn {i} ---")
        print(f"  User  : {user_text}")

        result = await process_input(sid, user_text)

        print(f"  Agent : {result['agent_text']}")
        print(f"  Score : {result['opportunity_score']}  Ended: {result['ended']}")

        if result.get("ended"):
            print()
            print("=" * 60)
            print("OUTCOME")
            print("=" * 60)
            outcome = result["outcome"]
            print(f"  Score       : {outcome['opportunity_score']} ({outcome['opportunity_label']})")
            print(f"  Next action : {outcome['recommended_next_action']}")
            print(f"  Summary     : {outcome['summary']}")
            print()
            print("  Decision Trace:")
            for t in outcome["decision_trace"]:
                src = f"[{t.get('extracted_source', '?')}/{t.get('wording_source', '?')}]"
                print(
                    f"    T{t['turn_index']}: {t['stage_before']}->{t['stage_after']} "
                    f"| {t['action']['type']} "
                    f"| score {t['score_before']}->{t['score_after']} "
                    f"{src}"
                )
            break

        print()

    print()
    print("Demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
