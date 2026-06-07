"""Entry point for Tongyi-DeepResearch via any OpenAI-compatible API.

Subclasses MultiTurnReactAgent from agent.py (Alibaba's verbatim code)
to override call_server for the configured API (DeepSeek, OpenRouter, etc.)
instead of local vLLM.
"""

import os
import random
import sys
import time
import tiktoken
from pathlib import Path
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

from settings import Config
from agent import MultiTurnReactAgent


# ---------------------------------------------------------------------------
# API-backed subclass — the only code change vs their original
# ---------------------------------------------------------------------------

class TongyiDeepResearchAgent(MultiTurnReactAgent):
    """MultiTurnReactAgent with call_server pointed at the configured API.

    Everything else (tool resolution, <tool_call>/<tool_response>/<answer>
    parsing, retry logic, stopping criteria) is unchanged from Alibaba's code.
    """

    def __init__(self, cfg: Config, **kwargs):
        self.cfg = cfg
        llm_cfg = {
            "model": cfg.model,
            "generate_cfg": {
                "max_input_tokens": 320000,
                "max_retries": 10,
                "temperature": 0.6,
                "top_p": 0.95,
                "presence_penalty": 1.1,
            },
            "model_type": "qwen_dashscope",
        }
        super().__init__(
            function_list=["search", "visit", "google_scholar"],
            llm=llm_cfg,
            **kwargs,
        )

    def call_server(self, msgs, planning_port=None, max_tries=10):
        """Chat completion via configured API (DeepSeek, OpenRouter, etc.).

        This is the exact swap documented in the Alibaba README section 6:
        https://github.com/Alibaba-NLP/DeepResearch
        """
        client = OpenAI(
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url,
            timeout=600.0,
        )

        base_sleep_time = 1
        for attempt in range(max_tries):
            try:
                print(
                    f"--- LLM call, attempt {attempt + 1}/{max_tries} ---"
                )
                chat_response = client.chat.completions.create(
                    model=self.cfg.model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get("temperature", 0.6),
                    top_p=self.llm_generate_cfg.get("top_p", 0.95),
                    logprobs=True,
                    max_tokens=10000,
                    presence_penalty=self.llm_generate_cfg.get(
                        "presence_penalty", 1.1
                    ),
                )
                content = chat_response.choices[0].message.content or ""

                # Prepend chain-of-thought reasoning if available (provider-specific)
                reasoning = getattr(
                    chat_response.choices[0].message, "reasoning", None
                )
                if reasoning:
                    reasoning = reasoning.strip()
                    content = (
                        f"<think>\n{reasoning}\n</think>\n" + content
                    )

                if content.strip():
                    print(
                        "--- LLM call successful, received response ---"
                    )
                    return content.strip()
                else:
                    print(
                        f"Warning: Attempt {attempt + 1} received an empty response."
                    )

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(
                    f"Error: Attempt {attempt + 1} failed with API/network error: {e}"
                )
            except Exception as e:
                print(
                    f"Error: Attempt {attempt + 1} failed with unexpected error: {e}"
                )

            if attempt < max_tries - 1:
                sleep_time = min(
                    base_sleep_time * (2**attempt) + random.uniform(0, 1),
                    30,
                )
                print(f"Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            else:
                print("Error: All retries exhausted.")

        return "LLM API server error!!!"

    def count_tokens(self, messages):
        """Override: use tiktoken instead of downloading the full HF tokenizer."""
        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(encoding.encode(content))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += len(encoding.encode(part.get("text", "")))
        return total


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def _empty_answer():
    """Placeholder answer field expected by their _run() data format."""
    return ""


def run(query: str, cfg: Config | None = None) -> str:
    """Run deep research on a query and return the final report.

    Args:
        query: Research question / topic.
        cfg: Config instance. Loaded from environment if not provided.

    Returns:
        The model's final answer (report text extracted from <answer> tags).
    """
    if cfg is None:
        cfg = Config.from_env()

    # Set env vars the verbatim tool files expect
    if cfg.serper_key_id:
        os.environ.setdefault("SERPER_KEY_ID", cfg.serper_key_id)
    if cfg.jina_api_keys:
        os.environ.setdefault("JINA_API_KEYS", cfg.jina_api_keys)
    os.environ.setdefault("API_KEY", cfg.summary_api_key)
    os.environ.setdefault("API_BASE", cfg.summary_api_base)
    if cfg.summary_model:
        os.environ.setdefault("SUMMARY_MODEL_NAME", cfg.summary_model)
    os.environ.setdefault("MAX_LLM_CALL_PER_RUN", str(cfg.max_tool_rounds))

    agent = TongyiDeepResearchAgent(cfg)

    # Format data as their _run() expects
    task_data = {
        "item": {
            "question": query,
            "answer": _empty_answer(),
        },
        "planning_port": 80,  # unused by our override
    }

    result = agent._run(task_data, cfg.model)
    prediction = result.get("prediction", "")
    if prediction and prediction not in ("No answer found.", "[Failed]"):
        return prediction

    # Fallback: return the last assistant message
    for msg in reversed(result.get("messages", [])):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return "No answer produced."


def _sanitize_filename(text: str) -> str:
    """Create a filesystem-safe filename from the query."""
    keep = []
    for ch in text[:60]:
        if ch.isalnum() or ch in " -_":
            keep.append(ch)
    return "".join(keep).strip().replace(" ", "_") or "research_report"


def save_report(topic: str, content: str, cfg: Config) -> Path:
    """Save the final report to disk."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_sanitize_filename(topic)}_Report.md"
    path = out_dir / filename
    path.write_text(content, encoding="utf-8")
    print(f"\n✅ Report saved: {path}")
    return path


def main():
    cfg = Config.from_env()
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    try:
        if query:
            report = run(query, cfg)
            save_report(query, report, cfg)
            # Print a preview
            preview = report[:500]
            if len(report) > 500:
                preview += "\n... [truncated]"
            print(f"\n📄 Report preview:\n{preview}")
        else:
            print("\n🔬 Deep Research Mode (type 'q' to quit)\n")
            while True:
                query = input("Q: ").strip()
                if query.lower() == "q":
                    break
                if not query:
                    continue
                report = run(query, cfg)
                save_report(query, report, cfg)
                print(f"\n📄 Report:\n{report}")
    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        raise


if __name__ == "__main__":
    main()
