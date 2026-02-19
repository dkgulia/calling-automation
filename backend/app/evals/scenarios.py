"""
Evaluation scenarios — scripted prospect turns with expected outcomes.

Each scenario is a list of user utterances that simulate a complete call.
Assertions define what the final outcome should look like.

These run with FORCE_RULE_BASED=true for deterministic results.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    name: str
    description: str
    turns: list[str]
    # Assertions on the final outcome
    expected_label: str  # "Weak", "Medium", or "Strong"
    min_score: float = 0.0
    max_score: float = 10.0
    expected_filled_slots: list[str] = field(default_factory=list)
    expect_ended: bool = True


SCENARIOS: list[Scenario] = [
    Scenario(
        name="strong_lead",
        description=(
            "Cooperative prospect reveals high pain, decent size, "
            "authority, budget, and timeline. Should score Strong."
        ),
        turns=[
            "Yeah sure, I have a minute. What's this about?",
            "Honestly, outbound is a huge pain point for us. "
            "We're spending way too much time on manual work. Pain is like 8 out of 10.",
            "We're about 50 people on the sales team.",
            "Yes, I'm the VP of Sales, I make the call on tools like this.",
            "We do have budget set aside for this quarter.",
            "We're looking to get something in place this quarter, ideally ASAP.",
            "Sure, I'd be open to a demo. Let's do it.",
        ],
        expected_label="Strong",
        min_score=7.0,
        max_score=10.0,
        expected_filled_slots=["pain", "company_size", "authority", "budget", "timeline"],
    ),
    Scenario(
        name="weak_lead",
        description=(
            "Prospect is not interested, gives minimal info, "
            "raises strong objection. Should score Weak."
        ),
        turns=[
            "I'm really not interested, we already have a tool for this.",
            "Look, I'm really busy right now, can you just send me an email?",
            "No thanks, goodbye.",
        ],
        expected_label="Weak",
        min_score=0.0,
        max_score=3.9,
        expected_filled_slots=[],
    ),
    Scenario(
        name="objection_heavy",
        description=(
            "Prospect shares some info but raises multiple objections. "
            "Should end up Weak — no authority, no budget, objection penalty."
        ),
        turns=[
            "Fine, I've got a minute.",
            "Our outbound is painful yeah, maybe a 6. We're about 30 people.",
            "We already have a tool actually, it's okay but not great.",
            "I'm not the decision maker, that would be my manager.",
            "I don't think there's budget for another tool right now.",
            "Maybe send me some info and I'll pass it along.",
        ],
        expected_label="Weak",
        min_score=0.0,
        max_score=3.9,
        expected_filled_slots=["pain", "company_size"],
    ),
]
