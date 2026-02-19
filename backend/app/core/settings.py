from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    deepgram_api_key: str = ""
    cartesia_api_key: str = ""
    cartesia_voice_id: str = "79a125e8-cd45-4c13-8a67-188112f4dd22"

    # DeepSeek R1 (OpenAI-compatible)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    llm_max_retries: int = 2
    llm_timeout_seconds: int = 12
    llm_min_confidence: float = 0.35

    # Silence timeout (Phase 5)
    silence_timeout_seconds: int = 30

    # Barge-in / interrupt handling (Phase 6)
    barge_in_min_words: int = 2
    barge_in_user_speech_timeout: float = 0.6

    # Force rule-based extraction + template wording (for eval determinism)
    force_rule_based: bool = False

    cors_origins: list[str] = ["http://localhost:5173"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
