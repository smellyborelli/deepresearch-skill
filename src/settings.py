"""Configuration for the Deep Research harness (generic OpenAI-compatible API)."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


@dataclass
class Config:
    """Runtime configuration loaded from environment."""

    # LLM provider (any OpenAI-compatible API)
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"

    # Search (Serper.dev)
    serper_key_id: str = ""

    # Page reading (Jina.ai)
    jina_api_keys: str = ""

    # Summarization LLM (used by visit.py for page extraction)
    summary_api_key: str = ""
    summary_api_base: str = ""
    summary_model: str = ""

    # Limits
    max_tokens: int = 32768
    max_tool_rounds: int = 50

    # Output
    output_dir: str = "output"

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.getenv("API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "API_KEY not set. Copy .env.example to .env and fill it in."
            )

        base_url = os.getenv("BASE_URL", "https://api.deepseek.com/v1")
        model = os.getenv("MODEL", "deepseek-chat")
        summary_api_key = os.getenv("SUMMARY_API_KEY", api_key)
        summary_api_base = os.getenv("SUMMARY_API_BASE", base_url)

        cfg = cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            serper_key_id=os.getenv("SERPER_KEY_ID", ""),
            jina_api_keys=os.getenv("JINA_API_KEYS", ""),
            summary_api_key=summary_api_key,
            summary_api_base=summary_api_base,
            summary_model=os.getenv("SUMMARY_MODEL_NAME", ""),
            max_tokens=int(os.getenv("MAX_TOKENS", "32768")),
            max_tool_rounds=int(os.getenv("MAX_TOOL_ROUNDS", "50")),
            output_dir=os.getenv("OUTPUT_DIR", "output"),
        )

        # Set env vars the verbatim Alibaba tool files expect
        os.environ.setdefault("API_KEY", cfg.summary_api_key)
        os.environ.setdefault("API_BASE", cfg.summary_api_base)
        if cfg.summary_model:
            os.environ.setdefault("SUMMARY_MODEL_NAME", cfg.summary_model)
        if cfg.serper_key_id:
            os.environ.setdefault("SERPER_KEY_ID", cfg.serper_key_id)
        if cfg.jina_api_keys:
            os.environ.setdefault("JINA_API_KEYS", cfg.jina_api_keys)

        return cfg
