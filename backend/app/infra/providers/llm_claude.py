# TODO: Replace with real Anthropic Claude API integration.
#
# In production this provider will:
#   1. Accept the conversation transcript + system prompt
#   2. Call Claude via the Anthropic SDK to:
#      a. Generate the next AI sales-rep utterance (response generation)
#      b. Extract structured fields from prospect replies (extraction)
#   3. Return structured output to the domain layer
#
# Integration point for Pipecat's LLMService:
#   pipeline.add(ClaudeLLMService(api_key=settings.anthropic_api_key))


async def generate_response(transcript: list[dict], system_prompt: str) -> str:
    """Stub: returns a canned sales-rep reply."""
    return "That's great to hear. Could you tell me more about your current workflow and where the biggest bottlenecks are?"


async def extract_fields(transcript: list[dict]) -> dict:
    """Stub: returns fake extracted fields as if Claude parsed the conversation."""
    return {
        "company_size": "50-200 employees",
        "pain": "Manual outbound process taking 4+ hours/day",
        "budget": "$20k-50k annual",
        "authority": "VP of Sales, direct decision-maker",
        "timeline": "Looking to implement within Q2",
    }
