from fastapi import APIRouter, HTTPException

from app.infra.session_store import get_session
from app.schemas.simulation import (
    InputRequest,
    RunRequest,
    RunResponse,
    TurnResponse,
)
from app.schemas.outcome import OutcomeResponse
from app.usecases.run_simulation import start_session
from app.usecases.process_input import process_input
from app.usecases.generate_prospect_turn import generate_prospect_turn

router = APIRouter(prefix="/api/v1")


@router.post("/run", response_model=RunResponse)
async def post_run(body: RunRequest | None = None):
    mode = body.prospect_mode if body else "human"
    result = start_session(prospect_mode=mode)
    return RunResponse(**result)


@router.post("/input/{session_id}", response_model=TurnResponse)
async def post_input(session_id: str, body: InputRequest):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await process_input(session_id, body.user_text)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return TurnResponse(**result)


@router.post("/prospect/{session_id}", response_model=TurnResponse)
async def post_prospect_turn(session_id: str):
    """Generate an AI prospect turn and process it."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("prospect_mode") != "ai":
        raise HTTPException(
            status_code=400, detail="Session is not in AI prospect mode"
        )

    result = await generate_prospect_turn(session_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return TurnResponse(**result)


@router.get("/outcome/{session_id}", response_model=OutcomeResponse)
async def get_outcome(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session["status"] == "running":
        return OutcomeResponse(status="running", outcome=None)

    return OutcomeResponse(status="completed", outcome=session["outcome"])
