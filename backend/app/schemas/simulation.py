from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Literal


class RunRequest(BaseModel):
    prospect_mode: Literal["human", "ai"] = "human"


class RunResponse(BaseModel):
    session_id: str
    status: str
    agent_text: str
    prospect_mode: str = "human"
    connect_info: dict


class InputRequest(BaseModel):
    user_text: str


class TurnResponse(BaseModel):
    status: str
    agent_text: str | None = None
    prospect_text: str | None = None
    opportunity_score: float | None = None
    ended: bool = False
    outcome: dict[str, Any] | None = None
