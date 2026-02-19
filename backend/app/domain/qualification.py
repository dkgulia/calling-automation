"""
Qualification scoring — explicit, explainable, and deterministic.

Every point added or subtracted is tagged with a reason so the trace
can explain exactly why a prospect scored the way they did.

Scoring model (additive with clamp to 0-10):
  +2  pain >= 7           (strong pain signal)
  +1  pain 4-6            (moderate pain)
  +1  company_size >= 20  (meaningful account)
  +2  authority is True   (talking to decision-maker)
  +2  budget is True      (budget available)
  +1  timeline present    (not "unknown")
  -1  objection: already_have_tool / not_interested / too_expensive
  -0.5 objection: send_email / busy  (total capped at -2)

Design note: the score is re-computed from scratch each turn (not
incrementally) so it's always consistent with the current state.
"""

from __future__ import annotations

from app.domain.state import ExtractedSignals, ProspectState

# Objections that strongly indicate a weak lead
_STRONG_OBJECTIONS = {"already_have_tool", "not_interested", "too_expensive"}
_MILD_OBJECTIONS = {"send_email", "busy"}


def score_opportunity(state: ProspectState, signals: ExtractedSignals) -> float:
    """
    Compute opportunity score (0-10) from current state and latest signals.

    The score reflects how promising this prospect is based on everything
    we've learned so far.  It is always recomputed from the full state,
    not incrementally adjusted, to avoid drift.

    Returns the clamped score as a float.
    """
    score = 0.0

    # --- Pain ---
    pain = state.learned_fields.get("pain")
    if pain is not None:
        if pain >= 7:
            score += 2.0
        elif pain >= 4:
            score += 1.0

    # --- Company size ---
    size = state.learned_fields.get("company_size")
    if size is not None and size >= 20:
        score += 1.0

    # --- Authority ---
    authority = state.learned_fields.get("authority")
    if authority is True:
        score += 2.0

    # --- Budget ---
    budget = state.learned_fields.get("budget")
    if budget is True:
        score += 2.0

    # --- Timeline ---
    timeline = state.learned_fields.get("timeline")
    if timeline is not None and timeline != "unknown":
        score += 1.0

    # --- Objection penalties (capped at -2 total) ---
    objection_penalty = 0.0
    for obj in state.objections:
        if obj in _STRONG_OBJECTIONS:
            objection_penalty += 1.0
        elif obj in _MILD_OBJECTIONS:
            objection_penalty += 0.5
    score -= min(objection_penalty, 2.0)

    # --- Clamp to 0-10 ---
    return max(0.0, min(10.0, score))


def score_breakdown(state: ProspectState) -> list[dict]:
    """
    Return an itemized list of scoring contributions.

    Each item: {"field": str, "points": float, "reason": str}
    """
    items: list[dict] = []

    pain = state.learned_fields.get("pain")
    if pain is not None:
        if pain >= 7:
            items.append({"field": "pain", "points": 2.0, "reason": f"Strong pain signal ({pain}/10)"})
        elif pain >= 4:
            items.append({"field": "pain", "points": 1.0, "reason": f"Moderate pain ({pain}/10)"})
        else:
            items.append({"field": "pain", "points": 0.0, "reason": f"Low pain ({pain}/10)"})

    size = state.learned_fields.get("company_size")
    if size is not None:
        if size >= 20:
            items.append({"field": "company_size", "points": 1.0, "reason": f"Meaningful account ({size} employees)"})
        else:
            items.append({"field": "company_size", "points": 0.0, "reason": f"Small account ({size} employees)"})

    authority = state.learned_fields.get("authority")
    if authority is True:
        items.append({"field": "authority", "points": 2.0, "reason": "Decision-maker confirmed"})
    elif authority is False:
        items.append({"field": "authority", "points": 0.0, "reason": "Not a decision-maker"})

    budget = state.learned_fields.get("budget")
    if budget is True:
        items.append({"field": "budget", "points": 2.0, "reason": "Budget available"})
    elif budget is False:
        items.append({"field": "budget", "points": 0.0, "reason": "No budget"})

    timeline = state.learned_fields.get("timeline")
    if timeline is not None and timeline != "unknown":
        items.append({"field": "timeline", "points": 1.0, "reason": f"Timeline: {timeline}"})

    objection_penalty = 0.0
    obj_items: list[dict] = []
    for obj in state.objections:
        if obj in _STRONG_OBJECTIONS:
            obj_items.append({"field": "objection", "points": -1.0, "reason": f"Objection: {obj}"})
            objection_penalty += 1.0
        elif obj in _MILD_OBJECTIONS:
            obj_items.append({"field": "objection", "points": -0.5, "reason": f"Mild objection: {obj}"})
            objection_penalty += 0.5

    # Cap total penalty at -2
    if objection_penalty > 2.0 and obj_items:
        scale = 2.0 / objection_penalty
        for item in obj_items:
            item["points"] = round(item["points"] * scale, 1)

    items.extend(obj_items)

    return items


def score_opportunity_with_breakdown(
    state: ProspectState, signals: ExtractedSignals
) -> tuple[float, list[dict], str]:
    """
    Compute score and return (score, breakdown_items, explanation_text).
    """
    score = score_opportunity(state, signals)
    items = score_breakdown(state)
    label = label_from_score(score)

    positives = [i for i in items if i["points"] > 0]
    negatives = [i for i in items if i["points"] < 0]

    parts = [f"Score: {score:.1f}/10 ({label})."]
    if positives:
        reasons = "; ".join(i["reason"] for i in positives)
        parts.append(f"Positives: {reasons}.")
    if negatives:
        reasons = "; ".join(i["reason"] for i in negatives)
        parts.append(f"Deductions: {reasons}.")
    if not positives and not negatives:
        parts.append("No qualification signals gathered.")

    return score, items, " ".join(parts)


def label_from_score(score: float) -> str:
    """
    Categorize a numeric score into a human-readable label.

    Thresholds:
      Weak:   0.0 – 3.9
      Medium: 4.0 – 6.9
      Strong: 7.0 – 10.0
    """
    if score < 4.0:
        return "Weak"
    if score < 7.0:
        return "Medium"
    return "Strong"
