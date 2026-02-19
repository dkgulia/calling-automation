import type { RunResponse, OutcomeResponse, TurnResponse } from "../types/outcome";

const BASE = "";

export async function startSimulation(
  prospectMode: "human" | "ai" = "human"
): Promise<RunResponse> {
  const res = await fetch(`${BASE}/api/v1/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prospect_mode: prospectMode }),
  });
  if (!res.ok) throw new Error(`Run failed: ${res.status}`);
  return res.json();
}

export async function sendInput(
  sessionId: string,
  userText: string
): Promise<TurnResponse> {
  const res = await fetch(`${BASE}/api/v1/input/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_text: userText }),
  });
  if (!res.ok) throw new Error(`Input failed: ${res.status}`);
  return res.json();
}

export async function fetchProspectTurn(
  sessionId: string
): Promise<TurnResponse> {
  const res = await fetch(`${BASE}/api/v1/prospect/${sessionId}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`Prospect turn failed: ${res.status}`);
  return res.json();
}

export async function fetchOutcome(
  sessionId: string
): Promise<OutcomeResponse> {
  const res = await fetch(`${BASE}/api/v1/outcome/${sessionId}`);
  if (!res.ok) throw new Error(`Outcome fetch failed: ${res.status}`);
  return res.json();
}
