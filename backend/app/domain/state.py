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


# Extracted signals — output of analyzing one user utterance

@dataclass
class ExtractedSignals:

    intent: str = "answer"  # "answer" | "objection" | "end" | "off_topic"
    company_size: int | None = None
    pain: int | None = None
    budget: bool | None = None
    authority: bool | None = None
    timeline: str | None = None
    objection_type: str | None = None
    confidence: float = 0.5
    answered_slot: str | None = None
    is_correction: bool = False

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
            "answered_slot": self.answered_slot,
            "is_correction": self.is_correction,
        }


# Rule-based signal extractor (Phase 1 only)

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


_CORRECTION_PATTERNS: list[str] = [
    r"\bactually\b",
    r"\bsorry\b.*\b(?:meant|mean)\b",
    r"\bi mean\b",
    r"\bcorrection\b",
    r"\bno[, ]+it'?s\b",
    r"\blet me correct\b",
    r"\bwait[, ]+(?:it'?s|we'?re|i'?m)\b",
]


def extract_signals_rule_based(user_text: str) -> ExtractedSignals:

    text = user_text.lower().strip()
    signals = ExtractedSignals()

    # --- Detect correction intent ---
    for pattern in _CORRECTION_PATTERNS:
        if re.search(pattern, text):
            signals.is_correction = True
            break

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
    objection_counts: dict[str, int] = field(default_factory=dict)
    interest_score: float = 0.0  # starts at zero; evidence-based scoring only
    last_user_text: str | None = None
    last_agent_text: str | None = None
    last_asked_slot: str | None = None
    slot_confidences: dict[str, float] = field(default_factory=dict)
    slot_sources: dict[str, str] = field(default_factory=dict)

    def missing_slots(self) -> list[str]:
        """Return slot names that haven't been filled yet, in priority order."""
        return [s for s in SLOT_PRIORITY if self.learned_fields.get(s) is None]

    def filled_slots(self) -> list[str]:
        """Return slot names that have been filled."""
        return [s for s in QUALIFICATION_SLOTS if self.learned_fields.get(s) is not None]

    def update_from_signals(
        self,
        signals: ExtractedSignals,
        min_confidence: float = 0.0,
        extracted_source: str = "rule_based",
    ) -> None:
        meets_confidence = signals.confidence >= min_confidence

        if meets_confidence:
            slot_values: list[tuple[str, Any]] = [
                ("company_size", signals.company_size),
                ("pain", signals.pain),
                ("budget", signals.budget),
                ("authority", signals.authority),
                ("timeline", signals.timeline),
            ]

            # Count how many slots have explicit data — if 2+, the user gave
            # a substantive multi-info answer (not filler) and all should fill.
            explicit_count = sum(1 for _, v in slot_values if v is not None)

            for slot_name, new_val in slot_values:
                if new_val is None:
                    continue
                # Alignment gating: if we asked a specific slot, only accept
                # fills for that slot unless confidence is very high, or the
                # user provided multiple slot values (substantive answer).
                # Corrections always bypass alignment gating.
                if self.last_asked_slot is not None and not signals.is_correction:
                    if (
                        signals.answered_slot != self.last_asked_slot
                        and slot_name != self.last_asked_slot
                        and signals.confidence < 0.85
                        and explicit_count < 2
                    ):
                        continue

                current_val = self.learned_fields[slot_name]
                if current_val is None:
                    # Empty slot — fill it
                    self.learned_fields[slot_name] = new_val
                    self.slot_confidences[slot_name] = signals.confidence
                    self.slot_sources[slot_name] = extracted_source
                else:
                    # Already filled — allow overwrite only on correction
                    # or substantially higher confidence
                    prev_conf = self.slot_confidences.get(slot_name, 0.5)
                    allow = (
                        signals.is_correction
                        or (
                            signals.confidence >= (prev_conf + 0.25)
                            and prev_conf < 0.85
                        )
                    )
                    if allow:
                        self.learned_fields[slot_name] = new_val
                        self.slot_confidences[slot_name] = signals.confidence
                        self.slot_sources[slot_name] = extracted_source

        # Always track objections (unique list + counts)
        if signals.objection_type:
            self.objection_counts[signals.objection_type] = (
                self.objection_counts.get(signals.objection_type, 0) + 1
            )
            if signals.objection_type not in self.objections:
                self.objections.append(signals.objection_type)


# ---------------------------------------------------------------------------
# Action — what the agent should do next
# ---------------------------------------------------------------------------

@dataclass
class Action:

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
