"""
Domain state models for the cold-call simulation.

This module defines the core data structures that flow through the system:
- CallStage: finite state machine for conversation phases
- ProspectState: mutable accumulator for everything learned during a call
- ExtractedSignals: structured output from analyzing a single user utterance
- Action: what the agent should do next (decided by the decision engine)
- TraceTurn / DecisionTrace: full audit trail for explainability

Design tradeoff: we use dataclasses (not Pydantic) in the domain layer to keep
it framework-free. Pydantic models live in schemas/ for the HTTP boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Call stages — finite state machine
# ---------------------------------------------------------------------------

class CallStage(Enum):
    """
    Conversation phases.  Transitions are managed by the decision engine
    (see decision_engine.next_stage_from_action).

    INTRO      -> opening, rapport building
    DISCOVERY  -> open-ended probing (pain, workflow)
    QUALIFY    -> explicit BANT slot-filling
    OBJECTION  -> handling pushback
    CLOSE      -> asking for next step (demo, meeting)
    END        -> call is over
    """
    INTRO = "INTRO"
    DISCOVERY = "DISCOVERY"
    QUALIFY = "QUALIFY"
    OBJECTION = "OBJECTION"
    CLOSE = "CLOSE"
    END = "END"


# The five BANT+ qualification slots we try to fill.
QUALIFICATION_SLOTS = ("pain", "company_size", "authority", "budget", "timeline")

# Preferred probing order — pain first because it drives urgency,
# then company_size (quick factual), authority (who decides), budget, timeline.
SLOT_PRIORITY = ["pain", "company_size", "authority", "budget", "timeline"]


# ---------------------------------------------------------------------------
# Extracted signals — output of analyzing one user utterance
# ---------------------------------------------------------------------------

@dataclass
class ExtractedSignals:
    """
    Structured signals extracted from a single user utterance.

    In Phase 1 these come from rule-based keyword matching.
    In later phases, Claude will produce these via structured extraction.

    Fields:
        intent: high-level classification of what the user is doing
        company_size: employee count if mentioned (None = not mentioned)
        pain: pain severity 0-10 if expressed (None = not mentioned)
        budget: True if budget is available, False if rejected, None if unknown
        authority: True if speaker is decision-maker, None if unknown
        timeline: freeform timeline string if mentioned
        objection_type: specific objection category if intent == "objection"
        confidence: how confident we are in this extraction (0-1)
    """
    intent: str = "answer"  # "answer" | "objection" | "end" | "off_topic"
    company_size: int | None = None
    pain: int | None = None
    budget: bool | None = None
    authority: bool | None = None
    timeline: str | None = None
    objection_type: str | None = None
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "company_size": self.company_size,
            "pain": self.pain,
            "budget": self.budget,
            "authority": self.authority,
            "timeline": self.timeline,
            "objection_type": self.objection_type,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Rule-based signal extractor (Phase 1 only)
# ---------------------------------------------------------------------------

# Objection patterns: (regex, objection_type)
_OBJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"\bnot interested\b", "not_interested"),
    (r"\bno thanks\b", "not_interested"),
    (r"\bdon'?t need\b", "not_interested"),
    (r"\balready (?:use|have|got)\b", "already_have_tool"),
    (r"\bwe use\b", "already_have_tool"),
    (r"\btoo expensive\b", "too_expensive"),
    (r"\bcan'?t afford\b", "too_expensive"),
    (r"\bno budget\b", "too_expensive"),
    (r"\bsend (?:me |an )?(?:email|details|info)\b", "send_email"),
    (r"\bjust (?:email|send)\b", "send_email"),
    (r"\bbusy\b", "busy"),
    (r"\bbad time\b", "busy"),
    (r"\bcall (?:me )?back\b", "busy"),
]

# End-of-call signals
_END_PATTERNS: list[str] = [
    r"\bgoodbye\b",
    r"\bhang up\b",
    r"\bgotta go\b",
    r"\bend (?:the )?call\b",
]


def extract_signals_rule_based(user_text: str) -> ExtractedSignals:
    """
    Cheap, deterministic signal extraction using keyword heuristics.

    This is intentionally simple — the goal is a working pipeline that can be
    swapped for Claude-based extraction in Phase 2 without changing the
    downstream decision engine at all.

    Tradeoff: low recall on nuanced language, but zero latency and no API cost.
    """
    text = user_text.lower().strip()
    signals = ExtractedSignals()

    # --- Check for end-of-call intent ---
    for pattern in _END_PATTERNS:
        if re.search(pattern, text):
            signals.intent = "end"
            signals.confidence = 0.8
            return signals

    # --- Check for objections ---
    for pattern, objection_type in _OBJECTION_PATTERNS:
        if re.search(pattern, text):
            signals.intent = "objection"
            signals.objection_type = objection_type
            signals.confidence = 0.7
            # Don't return yet — we may still extract slots from the same turn
            break

    # --- Extract company_size (look for numbers near people/employee words) ---
    size_match = re.search(
        r"(\d+)\s*(?:people|employees|folks|team|person|engineers|devs)", text
    )
    if size_match:
        signals.company_size = int(size_match.group(1))
        signals.confidence = max(signals.confidence, 0.7)

    # --- Extract pain signals ---
    pain_keywords = {
        "struggling": 8, "painful": 8, "frustrated": 7, "waste": 7,
        "slow": 6, "manual": 6, "tedious": 6, "hours": 5,
        "annoying": 5, "problem": 5, "issue": 4, "challenge": 4,
    }
    max_pain = None
    for kw, score in pain_keywords.items():
        if kw in text:
            if max_pain is None or score > max_pain:
                max_pain = score
    if max_pain is not None:
        signals.pain = max_pain

    # --- Extract budget ---
    if re.search(r"\b(?:have|got|set aside|allocated)\b.*\bbudget\b", text):
        signals.budget = True
    elif re.search(r"\b(?:no budget|can'?t afford|too expensive)\b", text):
        signals.budget = False
    elif re.search(r"\bbudget\b.*\b(?:maybe|depends|later|next quarter)\b", text):
        signals.budget = False  # soft no
        signals.timeline = "uncertain"

    # --- Extract authority ---
    # Check delegation/negative signals first (before job titles) because
    # "my manager" contains "manager" which would falsely match the title pattern.
    if re.search(r"\b(?:need to ask|check with|my boss|my manager|not the decision)\b", text):
        signals.authority = False
    elif re.search(r"\bi (?:decide|approve|sign off|own|handle|make the call)\b", text):
        signals.authority = True
        signals.confidence = max(signals.confidence, 0.7)
    elif re.search(r"\b(?:vp|director|head of|cto|ceo|founder)\b", text):
        signals.authority = True

    # --- Extract timeline ---
    timeline_match = re.search(
        r"\b(?:this quarter|next quarter|q[1-4]|this month|next month"
        r"|this year|next year|asap|immediately|soon|later|no rush)\b",
        text,
    )
    if timeline_match:
        signals.timeline = timeline_match.group(0)

    # If any slot was extracted, ensure confidence meets minimum usability
    # for confidence-gated update_from_signals (Phase 4).
    has_slots = any([
        signals.company_size is not None,
        signals.pain is not None,
        signals.budget is not None,
        signals.authority is not None,
        signals.timeline is not None,
    ])
    if has_slots:
        signals.confidence = max(signals.confidence, 0.6)

    # If we found no real content and it's not an objection, check off-topic
    if (
        signals.intent == "answer"
        and not has_slots
    ):
        # Could be a greeting, filler, or off-topic
        if re.search(r"\b(?:hello|hi|hey|what'?s this|who is this)\b", text):
            signals.intent = "answer"  # greeting is still an answer
            signals.confidence = 0.6
        elif len(text.split()) <= 3:
            signals.intent = "off_topic"
            signals.confidence = 0.3

    return signals


# ---------------------------------------------------------------------------
# Prospect state — mutable accumulator for the whole call
# ---------------------------------------------------------------------------

@dataclass
class ProspectState:
    """
    Everything we know about the prospect and the call so far.

    This is the central state object that flows through:
      extract_signals -> score_opportunity -> decide_next_action -> build_outcome

    Design note: learned_fields uses a flat dict with None values instead of
    nested optionals, making it trivial to count filled slots and serialize.
    """
    session_id: str = ""
    stage: CallStage = CallStage.INTRO
    turn_count: int = 0
    learned_fields: dict[str, Any] = field(default_factory=lambda: {
        "company_size": None,
        "pain": None,
        "budget": None,
        "authority": None,
        "timeline": None,
    })
    objections: list[str] = field(default_factory=list)
    interest_score: float = 0.0  # starts at zero; evidence-based scoring only
    last_user_text: str | None = None
    last_agent_text: str | None = None

    def missing_slots(self) -> list[str]:
        """Return slot names that haven't been filled yet, in priority order."""
        return [s for s in SLOT_PRIORITY if self.learned_fields.get(s) is None]

    def filled_slots(self) -> list[str]:
        """Return slot names that have been filled."""
        return [s for s in QUALIFICATION_SLOTS if self.learned_fields.get(s) is not None]

    def update_from_signals(
        self, signals: ExtractedSignals, min_confidence: float = 0.0
    ) -> None:
        """
        Merge extracted signals into learned_fields.

        Confidence gating (Phase 4): a slot is only filled when:
          - the signal value is not None
          - the slot is currently None (never overwrite learned values)
          - signals.confidence >= min_confidence

        Objections are always appended (if not duplicate) regardless of
        confidence, because objection intent itself is the important signal.
        """
        meets_confidence = signals.confidence >= min_confidence

        if meets_confidence:
            if signals.company_size is not None and self.learned_fields["company_size"] is None:
                self.learned_fields["company_size"] = signals.company_size
            if signals.pain is not None and self.learned_fields["pain"] is None:
                self.learned_fields["pain"] = signals.pain
            if signals.budget is not None and self.learned_fields["budget"] is None:
                self.learned_fields["budget"] = signals.budget
            if signals.authority is not None and self.learned_fields["authority"] is None:
                self.learned_fields["authority"] = signals.authority
            if signals.timeline is not None and self.learned_fields["timeline"] is None:
                self.learned_fields["timeline"] = signals.timeline

        if signals.objection_type and signals.objection_type not in self.objections:
            self.objections.append(signals.objection_type)


# ---------------------------------------------------------------------------
# Action — what the agent should do next
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """
    Output of the decision engine: a concrete instruction for the agent.

    type: what kind of move to make
        "ASK_SLOT"          — probe for a specific qualification field
        "HANDLE_OBJECTION"  — address the prospect's pushback
        "CLOSE"             — attempt to book next step
        "END"               — wrap up the call

    slot: which field to probe (only set when type == "ASK_SLOT")
    message_goal: short human-readable description of the move's intent
    reason_codes: machine-readable list of reasons that led to this decision
    """
    type: str  # "ASK_SLOT" | "HANDLE_OBJECTION" | "CLOSE" | "END"
    slot: str | None = None
    message_goal: str = ""
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "slot": self.slot,
            "message_goal": self.message_goal,
            "reason_codes": self.reason_codes,
        }


# ---------------------------------------------------------------------------
# Decision trace — full audit trail
# ---------------------------------------------------------------------------

@dataclass
class TraceTurn:
    """One turn's worth of decision audit data."""
    turn_index: int = 0
    user_text: str | None = None
    agent_text: str | None = None
    extracted: dict = field(default_factory=dict)
    action: dict = field(default_factory=dict)
    score_before: float = 0.0
    score_after: float = 0.0
    reasons: list[str] = field(default_factory=list)
    stage_before: str = ""
    stage_after: str = ""
    extracted_source: str = "rule_based"  # "llm" | "rule_based"
    wording_source: str = "template"      # "llm" | "template"

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "user_text": self.user_text,
            "agent_text": self.agent_text,
            "extracted": self.extracted,
            "action": self.action,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "reasons": self.reasons,
            "stage_before": self.stage_before,
            "stage_after": self.stage_after,
            "extracted_source": self.extracted_source,
            "wording_source": self.wording_source,
        }


@dataclass
class DecisionTrace:
    """
    Accumulates TraceTurns across the whole call for post-call explainability.

    The trace answers "why did the agent do X on turn N?" by recording the
    extracted signals, computed score, chosen action, and reason codes at
    every step.
    """
    session_id: str = ""
    turns: list[TraceTurn] = field(default_factory=list)
    ended_reason: str | None = None

    def add_turn(
        self,
        *,
        turn_index: int,
        user_text: str | None,
        agent_text: str | None,
        extracted: ExtractedSignals,
        action: Action,
        score_before: float,
        score_after: float,
        stage_before: CallStage,
        stage_after: CallStage,
        extracted_source: str = "rule_based",
        wording_source: str = "template",
    ) -> None:
        """Append a fully-formed trace turn."""
        self.turns.append(TraceTurn(
            turn_index=turn_index,
            user_text=user_text,
            agent_text=agent_text,
            extracted=extracted.to_dict(),
            action=action.to_dict(),
            score_before=score_before,
            score_after=score_after,
            reasons=list(action.reason_codes),
            stage_before=stage_before.value,
            stage_after=stage_after.value,
            extracted_source=extracted_source,
            wording_source=wording_source,
        ))
