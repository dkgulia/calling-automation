"""
DeepSeek R1 provider — LLM extraction and wording via OpenAI-compatible API.

Uses AsyncOpenAI to call DeepSeek's API.  Two public functions:

  - extract_signals_llm:  structured signal extraction from user text
  - generate_agent_utterance_llm:  natural-language wording for a chosen action

Both raise on failure — callers MUST catch and fall back to rule-based / template.

reasoning_content returned by deepseek-reasoner is IGNORED entirely — we only
read choices[0].message.content (the final JSON answer).
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.core.settings import settings
from app.domain.state import Action, ExtractedSignals

logger = logging.getLogger("roister")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class DeepSeekR1Client:
    """Thin async wrapper around DeepSeek's OpenAI-compatible chat API."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=float(settings.llm_timeout_seconds),
        )

    async def chat_json(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0) -> dict:
        """Send a chat completion and parse the JSON response.

        Retries up to LLM_MAX_RETRIES on empty content.
        Ignores reasoning_content from deepseek-reasoner.
        """
        last_err: Exception | None = None
        for attempt in range(settings.llm_max_retries + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=settings.deepseek_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=float(settings.llm_timeout_seconds),
                )
                content = resp.choices[0].message.content
                if content and content.strip():
                    return json.loads(content)
                last_err = RuntimeError(
                    f"Empty content on attempt {attempt + 1}"
                )
                logger.warning("DeepSeek empty content (attempt %d/%d)",
                               attempt + 1, settings.llm_max_retries + 1)
            except json.JSONDecodeError as e:
                last_err = e
                logger.warning("DeepSeek JSON parse error (attempt %d/%d): %s",
                               attempt + 1, settings.llm_max_retries + 1, e)
            except Exception as e:
                last_err = e
                logger.warning("DeepSeek API error (attempt %d/%d): %s",
                               attempt + 1, settings.llm_max_retries + 1, e)
                if attempt >= settings.llm_max_retries:
                    raise
        raise last_err or RuntimeError(
            "DeepSeek returned empty content after retries"
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_client: DeepSeekR1Client | None = None


def _get_client() -> DeepSeekR1Client:
    global _client
    if _client is None:
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not configured")
        _client = DeepSeekR1Client()
    return _client


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
You are a signal extractor for a B2B sales cold-call. Output only valid json.

Extract from the user's latest message:
{"intent":"answer|objection|end|off_topic","company_size":null,"pain":null,\
"budget":null,"authority":null,"timeline":null,"objection_type":null,"confidence":0.5}

Field rules:
- intent: "answer" (sharing info), "objection" (pushback), "end" (goodbye/hangup), "off_topic"
- company_size: integer employee count or null
- pain: 0-10 severity or null (10 = extreme)
- budget: true (available), false (rejected/none), null (unknown)
- authority: true (decision-maker), false (not), null (unknown)
- timeline: short string ("this quarter","Q2","asap") or null
- objection_type: "not_interested"|"already_have_tool"|"too_expensive"|"send_email"|"busy"|"other" or null
- confidence: 0.0-1.0 overall extraction confidence

Example: {"intent":"answer","company_size":50,"pain":7,"budget":null,\
"authority":true,"timeline":"this quarter","objection_type":null,"confidence":0.85}"""


async def extract_signals_llm(
    session_id: str,
    user_text: str,
    state_snapshot: dict,
) -> ExtractedSignals:
    """Extract structured signals from user text using DeepSeek R1.

    Raises on any failure — caller must fall back to rule-based extraction.
    """
    client = _get_client()

    known = {
        k: v
        for k, v in state_snapshot.get("learned_fields", {}).items()
        if v is not None
    }
    context = (
        f"Stage: {state_snapshot.get('stage', '?')}, "
        f"Turn: {state_snapshot.get('turn_count', 0)}"
    )
    if known:
        context += f", Known: {json.dumps(known)}"

    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": f"Context: {context}\nUser said: \"{user_text}\""},
    ]

    data = await client.chat_json(messages, max_tokens=200)
    logger.debug("Session %s LLM extraction raw: %s", session_id, data)

    return ExtractedSignals(
        intent=str(data.get("intent", "answer")),
        company_size=_safe_int(data.get("company_size")),
        pain=_safe_int(data.get("pain")),
        budget=_safe_bool(data.get("budget")),
        authority=_safe_bool(data.get("authority")),
        timeline=(
            data.get("timeline")
            if isinstance(data.get("timeline"), str)
            else None
        ),
        objection_type=(
            data.get("objection_type")
            if isinstance(data.get("objection_type"), str)
            else None
        ),
        confidence=float(data.get("confidence", 0.5)),
    )


# ---------------------------------------------------------------------------
# Agent wording
# ---------------------------------------------------------------------------

_WORDING_SYSTEM = """\
You are Alex, a friendly SDR at Roister (outbound sales platform). \
Output json: {"response":"your reply here"}

Write ONE reply: 1-2 sentences, at most 1 question. Sound natural and conversational.

Rules by action type:
- ASK_SLOT: briefly acknowledge what they said, then ask about that ONE topic
- HANDLE_OBJECTION: acknowledge the concern, then ask one redirect question
- CLOSE: propose a demo or meeting, ask to confirm
- END: polite goodbye, no extra questions

CRITICAL: If you see a "PREVIOUS reply" in context, your new reply MUST be completely different wording. \
Never repeat or paraphrase your last reply. Vary your opening, phrasing, and question.

Never invent company details you don't know. No markdown. No bullet points.

Example: {"response":"That's really helpful context. Roughly how large is your team handling outbound?"}"""


async def generate_agent_utterance_llm(
    action: Action,
    state_snapshot: dict,
    signals_dict: dict,
) -> str:
    """Generate natural agent wording for the chosen action using DeepSeek R1.

    Raises on any failure — caller must fall back to template text.
    """
    client = _get_client()

    known = {
        k: v
        for k, v in state_snapshot.get("learned_fields", {}).items()
        if v is not None
    }

    parts = [
        f"Action: {action.type}",
        f"Goal: {action.message_goal}",
    ]
    if action.slot:
        parts.append(f"Slot to ask: {action.slot}")
    if known:
        parts.append(f"Known about prospect: {json.dumps(known)}")
    if signals_dict.get("objection_type"):
        parts.append(f"Objection: {signals_dict['objection_type']}")

    # Add last agent text so LLM avoids repeating itself
    last_agent = state_snapshot.get("last_agent_text")
    if last_agent:
        parts.append(f"Your PREVIOUS reply (DO NOT repeat this): \"{last_agent}\"")

    # Add what the prospect just said so the LLM can respond contextually
    last_user = state_snapshot.get("last_user_text")
    if last_user:
        parts.append(f'Prospect just said: "{last_user}"')

    # Add objections already handled so agent doesn't re-address them
    handled = state_snapshot.get("objections", [])
    if handled:
        parts.append(f"Objections already addressed: {', '.join(handled)}")

    messages = [
        {"role": "system", "content": _WORDING_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]

    data = await client.chat_json(messages, max_tokens=120, temperature=0.8)

    # Extract text from various possible JSON shapes
    text = (
        data.get("response")
        or data.get("text")
        or data.get("reply")
        or ""
    )
    if not text:
        # Try any string value longer than 10 chars
        for v in data.values():
            if isinstance(v, str) and len(v) > 10:
                text = v
                break

    if not text:
        raise RuntimeError("LLM returned empty wording")

    return text.strip()


# ---------------------------------------------------------------------------
# AI Prospect utterance generation
# ---------------------------------------------------------------------------

_PROSPECT_SYSTEM_TEMPLATE = """\
You are a B2B prospect on a cold call. Output json: {{"response":"your reply here"}}

RULES:
1. Answer questions when asked, but STAY IN CHARACTER — your persona defines how cooperative or difficult you are
2. Give concrete data when you share info (employee counts, pain levels 1-10, yes/no on budget)
3. Keep it to 1-2 sentences, sound natural
4. Never repeat something you already said (check "Already revealed")
5. If your persona says you're skeptical or busy, actually BE that — push back, give short answers, raise objections

Your persona: {persona}

Example: {{"response":"{example}"}}"""

# Diverse prospect personas — some cooperative, some difficult, some hostile
_PROSPECT_PERSONAS = [
    # --- STRONG LEADS (cooperative, high-value) ---
    {
        "persona": "Head of Sales at a 35-person SaaS startup. Pain: 7/10 — reps waste time on manual prospecting. You're the decision-maker. Budget approved for Q1. Timeline: ASAP. You're friendly and open to hearing more.",
        "example": "We're 35 people, mostly engineers and a small sales team of 8. Prospecting eats up way too much time.",
    },
    {
        "persona": "Founder/CEO of a 15-person B2B consultancy. Pain: 9/10 — doing everything manually, no process. You make all decisions. Budget limited but flexible. Timeline: immediately. You're desperate for a solution and eager to talk.",
        "example": "It's just 15 of us. I do half the outbound myself and it's brutal honestly.",
    },
    # --- MEDIUM LEADS (mixed signals) ---
    {
        "persona": "Director of Revenue Ops at a 120-person fintech. Pain: 5/10 — current tools work but clunky. Your VP makes final calls, not you. Budget exists but needs justification. Timeline: maybe next quarter. You answer questions but aren't excited.",
        "example": "We're about 120 people. Our current setup works okay, not amazing but not terrible.",
    },
    {
        "persona": "Head of Growth at a 80-person e-commerce company. Pain: 6/10 — outbound is growing but secondary to inbound. You co-decide with CTO. Budget needs approval from above. Timeline: this quarter if compelling. You're polite but noncommittal.",
        "example": "We're about 80 people. Outbound isn't really our main channel but we're exploring it.",
    },
    # --- WEAK LEADS (difficult, objection-heavy) ---
    {
        "persona": "Sales Manager at a 200-person enterprise company. Pain: 3/10 — things are fine. You DON'T make tool decisions, that's your CRO. No budget allocated. Timeline: none. You're SKEPTICAL and SHORT with answers. Push back on most questions. You already use Outreach and are happy with it.",
        "example": "Look, we already use Outreach and it works fine. What exactly do you want?",
    },
    {
        "persona": "Account Executive at a 25-person agency. Pain: 2/10 — you don't really have outbound problems. You're NOT a decision-maker at all. No budget. No timeline. You're BUSY and ANNOYED at getting a cold call. Give very short, dismissive answers. Try to end the call quickly.",
        "example": "I'm really busy right now. Can you just send me an email?",
    },
    {
        "persona": "Operations Lead at a 45-person logistics company. Pain: 4/10 — some manual work but manageable. You'd need to check with your boss on tools. Budget is tight, probably no. Timeline: not this year. You're POLITE but keep saying you're not the right person and already have a tool that's 'good enough'.",
        "example": "Hmm, we already have something for that actually. I'm not really the one who handles these decisions.",
    },
    {
        "persona": "VP of Sales at a 90-person healthcare tech firm. Pain: 8/10 — outbound is a mess. You ARE the decision-maker. Budget: yes but only after seeing ROI proof. Timeline: this quarter. However you're VERY skeptical of cold call vendors — you've been burned before. You keep asking tough questions like 'how is this different from X?' and 'what's the catch?'",
        "example": "Yeah outbound is a disaster for us. But honestly I've heard this pitch a hundred times. What makes you different?",
    },
]


def _pick_prospect_persona(session_id: str = "") -> dict:
    """Select a prospect persona deterministically by session ID.

    Same session always gets the same persona. Different sessions get
    different personas based on hash.
    """
    idx = hash(session_id) % len(_PROSPECT_PERSONAS)
    return _PROSPECT_PERSONAS[idx]


async def generate_prospect_utterance_llm(
    agent_text: str,
    state_snapshot: dict,
    session_id: str = "",
) -> str:
    """Generate a realistic prospect reply using DeepSeek R1.

    Raises on any failure — caller must fall back to scripted text.
    """
    client = _get_client()

    known = {
        k: v
        for k, v in state_snapshot.get("learned_fields", {}).items()
        if v is not None
    }

    # Deterministic persona per session — same session always gets same persona
    persona_info = _pick_prospect_persona(session_id)

    system_prompt = _PROSPECT_SYSTEM_TEMPLATE.format(
        persona=persona_info["persona"],
        example=persona_info["example"],
    )

    parts = [f"Agent just said: \"{agent_text}\""]
    parts.append(f"Turn: {state_snapshot.get('turn_count', 0)}")
    if known:
        parts.append(f"Already revealed: {json.dumps(known)}")
    objections = state_snapshot.get("objections", [])
    if objections:
        parts.append(f"Objections already raised (do NOT repeat these): {', '.join(objections)}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(parts)},
    ]

    data = await client.chat_json(messages, max_tokens=120, temperature=0.9)
    text = data.get("response") or data.get("text") or data.get("reply") or ""
    if not text:
        for v in data.values():
            if isinstance(v, str) and len(v) > 10:
                text = v
                break
    if not text:
        raise RuntimeError("LLM returned empty prospect utterance")
    return text.strip()


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------

def _safe_int(val: object) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_bool(val: object) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1")
    return None
