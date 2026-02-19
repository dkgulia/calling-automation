from __future__ import annotations

from pydantic import BaseModel
from typing import Any


class LearnedFieldsSchema(BaseModel):
    company_size: int | None = None
    pain: int | None = None
    budget: bool | None = None
    authority: bool | None = None
    timeline: str | None = None


class TraceTurnSchema(BaseModel):
    turn_index: int
    user_text: str | None = None
    agent_text: str | None = None
    extracted: dict[str, Any] = {}
    action: dict[str, Any] = {}
    score_before: float = 0.0
    score_after: float = 0.0
    reasons: list[str] = []
    stage_before: str = ""
    stage_after: str = ""
    extracted_source: str = "rule_based"
    wording_source: str = "template"


class ScoreBreakdownItem(BaseModel):
    field: str
    points: float
    reason: str


class OutcomeSchema(BaseModel):
    learned_fields: LearnedFieldsSchema
    opportunity_score: float
    opportunity_label: str
    recommended_next_action: str
    summary: str
    score_breakdown: list[ScoreBreakdownItem] = []
    score_explanation: str = ""
    decision_trace: list[TraceTurnSchema] = []


class OutcomeResponse(BaseModel):
    status: str
    outcome: OutcomeSchema | None = None
