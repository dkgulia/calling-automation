"""
Evaluation runner — executes scenarios and checks assertions.

Usage:
    cd backend
    FORCE_RULE_BASED=true python -m app.evals.run_evals

Runs each scenario through the turn pipeline (text mode, no voice),
checks that the final outcome matches expectations, and prints a report.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Force rule-based mode for deterministic results
os.environ.setdefault("FORCE_RULE_BASED", "true")

from app.evals.scenarios import SCENARIOS, Scenario
from app.usecases.run_simulation import start_session
from app.usecases.process_input import process_input
from app.infra.session_store import get_session


async def run_scenario(scenario: Scenario) -> tuple[bool, list[str]]:
    """
    Run a single scenario and return (passed, list_of_failure_messages).
    """
    failures: list[str] = []

    # Start session
    session_info = start_session()
    session_id = session_info["session_id"]

    # Play each turn
    last_result = None
    for i, user_text in enumerate(scenario.turns):
        last_result = await process_input(session_id, user_text)
        if last_result.get("ended"):
            break

    # Get final session state
    session = get_session(session_id)
    outcome = session["outcome"] if session else None

    # If call didn't end naturally, that's okay — check what we have
    if outcome is None and last_result:
        # Use last_result's score info
        score = last_result.get("opportunity_score", 0.0)
    elif outcome:
        score = outcome["opportunity_score"]
    else:
        failures.append("No outcome and no last_result")
        return False, failures

    # Check label
    if outcome:
        label = outcome["opportunity_label"]
        if label != scenario.expected_label:
            failures.append(
                f"Label: expected '{scenario.expected_label}', got '{label}'"
            )

    # Check score range
    if score < scenario.min_score:
        failures.append(
            f"Score too low: {score:.1f} < {scenario.min_score}"
        )
    if score > scenario.max_score:
        failures.append(
            f"Score too high: {score:.1f} > {scenario.max_score}"
        )

    # Check filled slots
    if outcome and scenario.expected_filled_slots:
        learned = outcome.get("learned_fields", {})
        for slot in scenario.expected_filled_slots:
            val = learned.get(slot)
            if val is None:
                failures.append(f"Expected slot '{slot}' to be filled, got None")

    # Check score_breakdown exists
    if outcome and "score_breakdown" not in outcome:
        failures.append("Missing score_breakdown in outcome")

    return len(failures) == 0, failures


async def main():
    print("=" * 60)
    print("Roister Cold-Call Evaluation Harness")
    print("=" * 60)
    print()

    passed_count = 0
    total = len(SCENARIOS)

    for scenario in SCENARIOS:
        print(f"--- {scenario.name}: {scenario.description}")

        ok, failures = await run_scenario(scenario)

        if ok:
            print(f"  PASS")
            passed_count += 1
        else:
            print(f"  FAIL:")
            for f in failures:
                print(f"    - {f}")
        print()

    print("=" * 60)
    print(f"Results: {passed_count}/{total} passed")
    print("=" * 60)

    if passed_count < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
