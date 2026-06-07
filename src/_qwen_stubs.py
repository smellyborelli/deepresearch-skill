"""Minimal stubs for qwen_agent classes used by the Alibaba tools and agent.

The Alibaba repo's code imports from qwen_agent but only uses simple base
classes and type annotations. The full qwen-agent package is a heavy framework
that pulls in dozens of unrelated dependencies (audio, Azure, etc.). These
stubs provide exactly what's needed with zero external deps.
"""

from typing import Any


class Message:
    """Minimal Message schema — just a data holder."""
    def __init__(self, role="user", content=""):
        self.role = role
        self.content = content


ASSISTANT = "assistant"
USER = "user"
SYSTEM = "system"
FUNCTION = "function"
ROLE = "role"
DEFAULT_SYSTEM_MESSAGE = Message(role="system", content="You are a helpful assistant.")


class BaseTool:
    """Minimal tool base class matching the qwen_agent interface."""
    name = ""
    description = ""
    parameters = {}

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def call(self, params: str | dict, **kwargs) -> str:
        raise NotImplementedError

    def _verify_json_format_args(self, params: str | dict) -> dict:
        import json
        if isinstance(params, str):
            return json.loads(params)
        return params


def register_tool(name, allow_overwrite=False):
    """No-op decorator — we instantiate tools directly."""
    def decorator(cls):
        cls.name = name
        return cls
    return decorator


class BaseToolWithFileAccess(BaseTool):
    """Stub for tools needing file access."""
    pass


class FnCallAgent:
    """Minimal agent base class — __init__ is never called by our subclass."""
    pass


class BaseChatModel:
    """Placeholder for type hints."""
    pass


def build_text_completion_prompt(*args, **kwargs):
    """No-op — unused in our code path."""
    return ""


def format_as_text_message(*args, **kwargs):
    """No-op — unused in our code path."""
    return ""


def merge_generate_cfgs(*args, **kwargs):
    """No-op — unused in our code path."""
    return {}
