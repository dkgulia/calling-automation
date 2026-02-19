export interface LearnedFields {
  company_size: string;
  pain: string;
  budget: string;
  authority: string;
  timeline: string;
}

export interface ScoreBreakdownItem {
  field: string;
  points: number;
  reason: string;
}

export interface Outcome {
  session_id: string;
  learned_fields: LearnedFields;
  opportunity_score: number;
  opportunity_label: "Weak" | "Medium" | "Strong";
  recommended_next_action: string;
  summary: string;
  score_breakdown: ScoreBreakdownItem[];
  score_explanation: string;
}

export interface OutcomeResponse {
  status: "running" | "completed";
  outcome: Outcome | null;
}

export interface RunResponse {
  session_id: string;
  status: string;
  agent_text: string;
  prospect_mode: "human" | "ai";
  connect_info: { ws_url: string };
}

export interface TurnResponse {
  status: string;
  agent_text: string | null;
  prospect_text: string | null;
  opportunity_score: number | null;
  ended: boolean;
  outcome: Outcome | null;
}
